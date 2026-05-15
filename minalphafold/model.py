import torch
import torch.utils.checkpoint as torch_checkpoint
import math
from typing import cast

from .evoformer import Evoformer, SimplicialEvoformer
from .structure_module import StructureModule
from .initialization import init_gate_linear, init_linear, zero_linear
from .embedders import InputEmbedder, TemplatePair, TemplatePointwiseAttention, ExtraMsaStack
from .heads import DistogramHead, PLDDTHead, MaskedMSAHead, TMScoreHead, ExperimentallyResolvedHead
from .simplex import simplex_boundary_metric_confidence_map, simplex_boundary_metric_recycling_bins
from .utils import recycling_distance_bin


def _rms_normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    denom = torch.sqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True).clamp_min(1e-6))
    return x / denom.to(dtype=x.dtype)


def _scalar_override_to_float(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().float().cpu().item())


def _uses_pre_triangle_simplex_update(
    block: SimplicialEvoformer,
    *,
    simplex_pre_triangle_update_scale_override: torch.Tensor | None,
    simplex_pre_triangle_single_update_scale_override: torch.Tensor | None,
    simplex_triangle_attention_bias_scale_override: torch.Tensor | None,
    simplex_triangle_attention_value_scale_override: torch.Tensor | None,
) -> bool:
    pair_scale = max(float(getattr(block, "simplex_pre_triangle_update_scale", 0.0)), 0.0)
    pair_override = _scalar_override_to_float(simplex_pre_triangle_update_scale_override)
    if pair_override is not None:
        pair_scale = max(pair_override, 0.0)

    configured_single_scale = float(getattr(block, "simplex_pre_triangle_single_update_scale", -1.0))
    single_scale = pair_scale if configured_single_scale < 0.0 else max(configured_single_scale, 0.0)
    single_override = _scalar_override_to_float(simplex_pre_triangle_single_update_scale_override)
    if single_override is not None:
        single_scale = max(single_override, 0.0)

    triangle_bias_scale = max(float(getattr(block, "simplex_triangle_attention_bias_scale", 0.0)), 0.0)
    triangle_bias_override = _scalar_override_to_float(simplex_triangle_attention_bias_scale_override)
    if triangle_bias_override is not None:
        triangle_bias_scale = max(triangle_bias_override, 0.0)

    triangle_value_scale = max(float(getattr(block, "simplex_triangle_attention_value_scale", 0.0)), 0.0)
    triangle_value_override = _scalar_override_to_float(simplex_triangle_attention_value_scale_override)
    if triangle_value_override is not None:
        triangle_value_scale = max(triangle_value_override, 0.0)
    return pair_scale > 0.0 or single_scale > 0.0 or triangle_bias_scale > 0.0 or triangle_value_scale > 0.0


class AlphaFold2(torch.nn.Module):
    """Top-level AlphaFold2 model (Algorithm 2).

    Wires together the pipeline described in Supplementary Figure 1 and
    Algorithm 2:

    1. **Input embedding** (Algorithm 3): ``target_feat``/``residue_index``/
       ``msa_feat`` → initial ``m_si`` and ``z_ij``.
    2. **Recycling embedding** (Algorithm 32): ``LayerNorm(m_1i^prev)`` and
       ``LayerNorm(z_ij^prev) + Linear(one_hot(d_ij^prev))`` injected back
       into the first row of ``m_si`` and into ``z_ij``.
    3. **Template embedding** (supplement 1.7.1): template torsion angles are
       concatenated onto ``m_si``; template pair features pass through
       ``TemplatePair`` and are pooled via ``TemplatePointwiseAttention``
       into ``z_ij``.
    4. **Extra MSA stack** (supplement 1.7.2): large-depth auxiliary MSA rep
       updates ``z_ij`` through a shallow Evoformer-like stack.
    5. **Evoformer trunk** (Algorithm 6): the main stack. ``s_i = Linear(m_1i)``
       (Algorithm 6 line 12) is computed after ensemble averaging here.
    6. **Structure Module** (Algorithm 20): iterative IPA + backbone update →
       all-atom coordinates, per-layer frame/torsion trajectories, and the
       post-IPA single representation used by pLDDT.
    7. **Auxiliary heads** (supplement 1.9): distogram (eq 41), masked MSA
       (eq 42), experimentally-resolved (eq 43), pLDDT (Algorithm 29), and
       TM-score / PAE (supplement 1.9.7).

    Recycling (supplement 1.10 / Algorithm 31): the forward pass runs
    ``n_cycles`` times, re-embedding the previous cycle's ``m_1i``, ``z_ij``,
    and pseudo-β positions. During training ``n_cycles`` is sampled uniformly
    from ``{1, ..., n_cycles}`` and only the final cycle carries gradients.

    Ensembling (supplement 1.11.2 / Algorithm 2 lines 4, 18, 20): the
    pre-Structure-Module pipeline is run ``n_ensemble`` times per cycle and
    the averaged ``m̂_1i`` and ``ẑ_ij`` feed the Structure Module and heads.
    Per the supplement, ensembling is only used at inference; training uses
    ``n_ensemble = 1``.
    """

    def __init__(self, config):
        super().__init__()
        self.use_simplicial_evoformer = bool(getattr(config, "use_simplicial_evoformer", False))
        self.simplex_structure_readout_scale = float(getattr(config, "simplex_structure_readout_scale", 0.0))
        self.simplex_structure_pair_readout_scale = float(
            getattr(config, "simplex_structure_pair_readout_scale", 0.0)
        )
        self.simplex_boundary_metric_recycling_scale = float(
            getattr(config, "simplex_boundary_metric_recycling_scale", 0.0)
        )
        self.simplex_boundary_cochain_recycling_scale = float(
            getattr(config, "simplex_boundary_cochain_recycling_scale", 0.0)
        )
        self.simplex_boundary_cochain_recycling_metric_gate_scale = float(
            getattr(config, "simplex_boundary_cochain_recycling_metric_gate_scale", 0.0)
        )
        simplex_every_n = max(int(getattr(config, "simplex_every_n_blocks", 1)), 1)
        if self.use_simplicial_evoformer:
            self.evoformer_blocks = torch.nn.ModuleList(
                [
                    SimplicialEvoformer(config, enable_simplex=((idx + 1) % simplex_every_n == 0))
                    for idx in range(config.num_evoformer)
                ]
            )
        else:
            self.evoformer_blocks = torch.nn.ModuleList([Evoformer(config) for _ in range(config.num_evoformer)])
        self.structure_model = StructureModule(config)

        self.input_embedder = InputEmbedder(config)

        # Recycling embedders (Algorithm 32): LN-only for single/pair reps; distance bins use a learned linear
        self.recycle_norm_s = torch.nn.LayerNorm(config.c_m)
        self.recycle_norm_z = torch.nn.LayerNorm(config.c_z)
        self.recycle_linear_d = torch.nn.Linear(15, config.c_z)
        init_linear(self.recycle_linear_d, init="default")

        # Project from MSA channel dim (c_m) to single rep dim (c_s)
        self.single_rep_proj = torch.nn.Linear(config.c_m, config.c_s)
        init_linear(self.single_rep_proj, init="default")

        # Template processing
        self.template_pair_feat_linear = torch.nn.Linear(88, config.c_t)
        self.template_pair_stack = TemplatePair(config)
        self.template_pointwise_att = TemplatePointwiseAttention(config)
        init_linear(self.template_pair_feat_linear, init="relu")

        self.template_angle_linear_1 = torch.nn.Linear(57, config.c_m)
        self.template_angle_linear_2 = torch.nn.Linear(config.c_m, config.c_m)
        init_linear(self.template_angle_linear_1, init="relu")
        init_linear(self.template_angle_linear_2, init="relu")

        # Extra MSA processing
        self.extra_msa_feat_linear = torch.nn.Linear(25, config.c_e)
        self.extra_msa_blocks = torch.nn.ModuleList(
            [ExtraMsaStack(config) for _ in range(config.num_extra_msa)]
        )
        init_linear(self.extra_msa_feat_linear, init="default")

        # Prediction heads
        self.distogram_head = DistogramHead(config)
        self.plddt_head = PLDDTHead(config)
        self.masked_msa_head = MaskedMSAHead(config)
        self.tm_score_head = TMScoreHead(config)
        self.experimentally_resolved_head = ExperimentallyResolvedHead(config)

        self.config = config
        self._initialize_alphafold_parameters()

    @staticmethod
    def _zero_linear(linear: torch.nn.Linear):
        zero_linear(linear)

    @staticmethod
    def _init_gate_linear(linear: torch.nn.Linear):
        init_gate_linear(linear)

    def _initialize_alphafold_parameters(self):
        zero_init = bool(getattr(self.config, "zero_init", True))
        output_zero_init_classes = {
            "MSARowAttentionWithPairBias",
            "MSAColumnAttention",
            "MSAColumnGlobalAttention",
            "TemplatePointwiseAttention",
            "ExtraMsaStack",
            "TriangleAttentionStartingNode",
            "TriangleAttentionEndingNode",
            "InvariantPointAttention",
        }
        transition_zero_init_classes = {"MSATransition", "PairTransition"}

        for module in self.modules():
            class_name = module.__class__.__name__

            if hasattr(module, "linear_gate") and isinstance(module.linear_gate, torch.nn.Linear):
                self._init_gate_linear(module.linear_gate)

            if zero_init and class_name in output_zero_init_classes and hasattr(module, "linear_output"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_output))

            if zero_init and class_name in transition_zero_init_classes and hasattr(module, "linear_down"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_down))

            if zero_init and class_name == "OuterProductMean" and hasattr(module, "linear_out"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_out))

            if class_name in {"TriangleMultiplicationOutgoing", "TriangleMultiplicationIncoming"}:
                self._init_gate_linear(cast(torch.nn.Linear, module.gate1))
                self._init_gate_linear(cast(torch.nn.Linear, module.gate2))
                self._init_gate_linear(cast(torch.nn.Linear, module.gate))
                self._zero_linear(cast(torch.nn.Linear, module.out_linear))

            if zero_init and class_name == "StructureModule":
                self._zero_linear(cast(torch.nn.Linear, module.transition_linear_3))

            if zero_init and class_name == "BackboneUpdate":
                self._zero_linear(cast(torch.nn.Linear, module.linear))

            if zero_init and class_name == "AngleResnetBlock" and hasattr(module, "linear_2"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_2))

            if class_name == "InvariantPointAttention":
                # Set head weights so softplus(head_weights) = 1 at init.
                cast(torch.nn.Parameter, module.head_weights).data.fill_(math.log(math.e - 1.0))

    @staticmethod
    def _sampled_feature_slice(
        tensor: torch.Tensor,
        cycle_index: int,
        ensemble_index: int,
        *,
        base_ndim: int,
    ) -> torch.Tensor:
        """Index into a feature tensor by recycling cycle and ensemble sample.

        Supplement 1.11.2 / Algorithm 2 draws ``N_cycle × N_ensemble`` random
        samples of ``msa_feat``/``extra_msa_feat``. The data pipeline may
        materialise those samples ahead of time by prepending ``[cycle,
        ensemble, ...]`` axes to the feature, or just a ``[cycle, ...]`` axis,
        or neither. ``base_ndim`` is the ndim of a *single* sample (e.g. 4 for
        ``msa_feat = (batch, N_seq, N_res, 49)``); this helper strips whichever
        outer sampling axes are present so the model always sees one slice.
        """
        if tensor.ndim == base_ndim + 2:
            return tensor[cycle_index, ensemble_index]
        if tensor.ndim == base_ndim + 1:
            return tensor[cycle_index]
        return tensor

    def forward(
            self,
            target_feat: torch.Tensor,
            residue_index: torch.Tensor,
            msa_feat: torch.Tensor,
            extra_msa_feat: torch.Tensor,
            template_pair_feat: torch.Tensor,
            aatype: torch.Tensor,
            template_angle_feat: torch.Tensor | None = None,
            template_mask: torch.Tensor | None = None,
            template_residue_mask: torch.Tensor | None = None,
            seq_mask: torch.Tensor | None = None,
            msa_mask: torch.Tensor | None = None,
            extra_msa_mask: torch.Tensor | None = None,
            n_cycles: int = 3,
            n_ensemble: int = 1,
            detach_rotations: bool = True,
            simplex_teacher_ca_coords: torch.Tensor | None = None,
            simplex_teacher_ca_mask: torch.Tensor | None = None,
            simplex_teacher_forcing_weight: torch.Tensor | None = None,
            simplex_pair_update_scale_override: torch.Tensor | None = None,
            simplex_single_update_scale_override: torch.Tensor | None = None,
            simplex_outer_edge_context_scale_override: torch.Tensor | None = None,
            simplex_hodge_face_update_scale_override: torch.Tensor | None = None,
            simplex_edge_frame_message_scale_override: torch.Tensor | None = None,
            simplex_boundary_edge_frame_gate_scale_override: torch.Tensor | None = None,
            simplex_boundary_readout_directionality_override: torch.Tensor | None = None,
            simplex_boundary_hodge_readout_scale_override: torch.Tensor | None = None,
            simplex_boundary_edge_star_readout_scale_override: torch.Tensor | None = None,
            simplex_boundary_edge_star_residual_scale_override: torch.Tensor | None = None,
            simplex_vertex_star_context_scale_override: torch.Tensor | None = None,
            simplex_edge_star_context_scale_override: torch.Tensor | None = None,
            simplex_pre_triangle_update_scale_override: torch.Tensor | None = None,
            simplex_pre_triangle_single_update_scale_override: torch.Tensor | None = None,
            simplex_segment_cell_scale_override: torch.Tensor | None = None,
            simplex_msa_feedback_scale_override: torch.Tensor | None = None,
            simplex_boundary_pair_feedback_scale_override: torch.Tensor | None = None,
            simplex_boundary_pair_gate_scale_override: torch.Tensor | None = None,
            simplex_boundary_metric_gate_scale_override: torch.Tensor | None = None,
            simplex_triangle_attention_bias_scale_override: torch.Tensor | None = None,
            simplex_triangle_attention_value_scale_override: torch.Tensor | None = None,
            simplex_boundary_metric_recycling_scale_override: torch.Tensor | None = None,
            simplex_boundary_cochain_recycling_scale_override: torch.Tensor | None = None,
            simplex_local_neighbor_k_override: torch.Tensor | None = None,
            simplex_geometry_distance_weight_override: torch.Tensor | None = None,
            simplex_face_top_k_override: torch.Tensor | None = None,
            simplex_tetra_top_k_override: torch.Tensor | None = None,
            simplex_cell_score_outer_edge_weight_override: torch.Tensor | None = None,
        ):
        """Algorithm 2 forward pass. See the class docstring for the full map."""
        # seq_mask: (batch, N_res) — 1 for valid residues, 0 for padding
        # msa_mask: (batch, N_seq, N_res) — 1 for valid, 0 for padding
        # extra_msa_mask: (batch, N_extra, N_res) — 1 for valid, 0 for padding
        assert n_ensemble > 0
        assert n_cycles > 0

        if self.training:
            # Algorithm 31 line 1: N' ~ Uniform(1, N_cycle). Only iteration
            # N' carries gradients; earlier iterations are stop-grad'd via the
            # detach() calls at the bottom of the loop (Algorithm 31 line 4).
            n_cycles = int(torch.randint(1, n_cycles + 1, (1,), device=target_feat.device).item())
        self.last_n_cycles = int(n_cycles)
        self.last_n_ensemble = int(n_ensemble)
        simplex_boundary_metric_recycling_scale = self.simplex_boundary_metric_recycling_scale
        if simplex_boundary_metric_recycling_scale_override is not None:
            simplex_boundary_metric_recycling_scale = max(
                float(simplex_boundary_metric_recycling_scale_override.detach().float().cpu().item()),
                0.0,
            )
        simplex_boundary_cochain_recycling_scale = self.simplex_boundary_cochain_recycling_scale
        if simplex_boundary_cochain_recycling_scale_override is not None:
            simplex_boundary_cochain_recycling_scale = max(
                float(simplex_boundary_cochain_recycling_scale_override.detach().float().cpu().item()),
                0.0,
            )

        outer_grad = torch.is_grad_enabled()

        N_res = target_feat.shape[1]
        c_m = self.config.c_m
        c_z = self.config.c_z
        batch_size = target_feat.shape[0]

        # Default masks: all ones (no padding).
        if seq_mask is None:
            seq_mask = target_feat.new_ones(batch_size, N_res)
        assert seq_mask is not None
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]  # (batch, N_res, N_res)
        if msa_mask is None:
            msa_mask = target_feat.new_ones(batch_size, msa_feat.shape[1], N_res)
        assert msa_mask is not None
        if extra_msa_mask is None:
            extra_msa_mask = target_feat.new_ones(batch_size, extra_msa_feat.shape[1], N_res)
        assert extra_msa_mask is not None

        # Algorithm 2 line 1: m̂_1i^prev, ẑ_ij^prev, x̄_i^{prev,Cβ} ← 0, 0, 0.
        single_rep_prev = torch.zeros(batch_size, N_res, c_m, device=msa_feat.device)
        z_prev = torch.zeros(batch_size, N_res, N_res, c_z, device=msa_feat.device)
        x_prev = torch.zeros(batch_size, N_res, 3, device=msa_feat.device)
        ca_prev: torch.Tensor | None = None
        rotations_prev: torch.Tensor | None = None

        # Algorithm 2 line 2: recycling loop.
        for i in range(n_cycles):
            is_last = (i == n_cycles - 1)

            with torch.set_grad_enabled(is_last and outer_grad):
                # Algorithm 2 line 3: zero the ensemble accumulators. Per line 18
                # we accumulate only m_1i (the first MSA row) and z_ij — the
                # single rep s_i = Linear(m_1i) is obtained after averaging via
                # single_rep_proj, which is equivalent by linearity. The full
                # MSA representation is NOT averaged (Algorithm 2 line 18 only
                # averages m_1i, not m_si); the masked MSA head instead consumes
                # the last ensemble sample's full rep. In practice this only
                # matters at inference — supplement 1.11.2 uses N_ensemble = 1
                # at training, and the masked MSA head is training-only.
                single_rep_accum = torch.zeros(batch_size, N_res, c_m, device=msa_feat.device)
                structure_single_accum = torch.zeros(batch_size, N_res, self.config.c_s, device=msa_feat.device)
                pair_repr_accum = torch.zeros(batch_size, N_res, N_res, c_z, device=msa_feat.device)
                simplex_structure_single_accum = torch.zeros_like(structure_single_accum)
                simplex_structure_pair_accum = torch.zeros_like(pair_repr_accum)
                msa_repr_last: torch.Tensor | None = None
                simplex_aux_last: dict[str, torch.Tensor] | None = None

                # Algorithm 2 line 4: ensemble loop.
                for ensemble_index in range(n_ensemble):
                    simplex_aux_current: dict[str, torch.Tensor] | None = None
                    msa_feat_current = self._sampled_feature_slice(msa_feat, i, ensemble_index, base_ndim=4)
                    extra_msa_feat_current = self._sampled_feature_slice(extra_msa_feat, i, ensemble_index, base_ndim=4)
                    msa_mask_current = self._sampled_feature_slice(msa_mask, i, ensemble_index, base_ndim=3)
                    extra_msa_mask_current = self._sampled_feature_slice(extra_msa_mask, i, ensemble_index, base_ndim=3)

                    # Algorithm 2 line 5 (= Algorithm 3 / InputEmbedder).
                    msa_representation, pair_representation = self.input_embedder(
                        target_feat,
                        residue_index,
                        msa_feat_current,
                    )

                    # Algorithm 2 line 6 (= Algorithm 32 / RecyclingEmbedder):
                    #   m_1i  += LayerNorm(m_1i^prev)
                    #   z_ij  += LayerNorm(z_ij^prev) + Linear(one_hot(d_ij^prev))
                    # On the first cycle the prev tensors are zero, so these
                    # additions vanish. Clone before the in-place write to the
                    # first MSA row so the embedder's output tensor is untouched.
                    msa_repr = msa_representation.clone()
                    pair_repr = pair_representation.clone()
                    msa_repr[:, 0, :, :] += self.recycle_norm_s(single_rep_prev)
                    pair_repr += self.recycle_norm_z(z_prev)
                    pair_repr += self.recycle_linear_d(recycling_distance_bin(x_prev, n_bins=15))

                    # Algorithm 2 lines 7-8: template torsion-angle embedding,
                    # concatenated onto the MSA representation as extra rows so
                    # the Evoformer attends over real MSA + template angles
                    # uniformly. msa_mask is extended to cover the new rows.
                    evo_msa_mask = msa_mask_current
                    if template_angle_feat is not None and template_angle_feat.shape[1] > 0:
                        template_angle_repr = self.template_angle_linear_2(
                            torch.relu(self.template_angle_linear_1(template_angle_feat))
                        )
                        msa_repr = torch.cat([msa_repr, template_angle_repr], dim=1)
                        n_templ = template_angle_repr.shape[1]
                        if template_mask is not None:
                            templ_mask = template_mask[:, :, None].to(msa_mask_current.dtype) * seq_mask[:, None, :]
                        else:
                            templ_mask = msa_mask_current.new_ones(batch_size, n_templ, N_res)
                        evo_msa_mask = torch.cat([msa_mask_current, templ_mask], dim=1)

                    # Algorithm 2 lines 9-13: template pair stack + pointwise
                    # attention pool into z_ij. Template pair/angle paths touch
                    # disjoint tensors (z vs m), so the two blocks commute;
                    # we follow the supplement's ordering.
                    if template_pair_feat.shape[1] > 0:
                        template_pair_mask = None
                        if template_residue_mask is not None:
                            template_pair_mask = (
                                template_residue_mask[:, :, :, None]
                                * template_residue_mask[:, :, None, :]
                            )
                        template_pair = self.template_pair_feat_linear(template_pair_feat)
                        template_pair = self.template_pair_stack(template_pair, pair_mask=template_pair_mask)
                        pair_repr = pair_repr + self.template_pointwise_att(
                            template_pair,
                            pair_repr,
                            template_mask=template_mask,
                        )

                    # Algorithm 2 lines 14-16: extra MSA stack updates z_ij
                    # (shallow Evoformer-like blocks, supplement 1.7.2).
                    #
                    # Supplement 1.11.8: "we store the activations that are
                    # passed between the N_block = 48 Evoformer blocks.
                    # During the backward pass, we recompute all activations
                    # within the blocks." That's exactly what
                    # torch.utils.checkpoint does. We apply it only when
                    # gradients are required — during eval the checkpointed
                    # path would just add overhead for no memory benefit.
                    if extra_msa_feat_current.shape[1] > 0:
                        extra_msa_repr = self.extra_msa_feat_linear(extra_msa_feat_current)
                        for extra_block in self.extra_msa_blocks:
                            if self.training:
                                # ``checkpoint``'s type stub is ``Any | None`` so the
                                # tuple unpack fails type-check without the cast.
                                extra_msa_repr, pair_repr = cast(
                                    tuple[torch.Tensor, torch.Tensor],
                                    torch_checkpoint.checkpoint(
                                        extra_block,
                                        extra_msa_repr, pair_repr,
                                        extra_msa_mask=extra_msa_mask_current, pair_mask=pair_mask,
                                        use_reentrant=False,
                                    ),
                                )
                            else:
                                extra_msa_repr, pair_repr = extra_block(
                                    extra_msa_repr, pair_repr,
                                    extra_msa_mask=extra_msa_mask_current, pair_mask=pair_mask,
                                )

                    # Algorithm 2 line 17 (= Algorithm 6 / EvoformerStack).
                    single_repr = self.single_rep_proj(msa_repr[:, 0, :, :])
                    for block in self.evoformer_blocks:
                        if isinstance(block, SimplicialEvoformer):
                            use_checkpoint = self.training and not _uses_pre_triangle_simplex_update(
                                block,
                                simplex_pre_triangle_update_scale_override=(
                                    simplex_pre_triangle_update_scale_override
                                ),
                                simplex_pre_triangle_single_update_scale_override=(
                                    simplex_pre_triangle_single_update_scale_override
                                ),
                                simplex_triangle_attention_bias_scale_override=(
                                    simplex_triangle_attention_bias_scale_override
                                ),
                                simplex_triangle_attention_value_scale_override=(
                                    simplex_triangle_attention_value_scale_override
                                ),
                            )
                            if use_checkpoint:
                                # Non-reentrant checkpointing supports kwargs
                                # and nested outputs, so the simplex auxiliary
                                # dictionary can be recomputed just like the
                                # tensor activations inside the block. The
                                # pre-triangle simplex pass is excluded because
                                # its selected complex can contain variable-size
                                # packed tensors; eager execution avoids
                                # checkpoint recomputation metadata mismatches.
                                msa_repr, pair_repr, single_repr, simplex_aux = cast(
                                    tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
                                    torch_checkpoint.checkpoint(
                                        block,
                                        msa_repr,
                                        pair_repr,
                                        single_repr,
                                        msa_mask=evo_msa_mask,
                                        pair_mask=pair_mask,
                                        seq_mask=seq_mask,
                                        recycled_ca_coords=ca_prev,
                                        recycled_frames=rotations_prev,
                                        simplex_teacher_ca_coords=simplex_teacher_ca_coords,
                                        simplex_teacher_ca_mask=simplex_teacher_ca_mask,
                                        simplex_teacher_forcing_weight=simplex_teacher_forcing_weight,
                                        simplex_pair_update_scale_override=simplex_pair_update_scale_override,
                                        simplex_single_update_scale_override=simplex_single_update_scale_override,
                                        simplex_outer_edge_context_scale_override=(
                                            simplex_outer_edge_context_scale_override
                                        ),
                                        simplex_hodge_face_update_scale_override=(
                                            simplex_hodge_face_update_scale_override
                                        ),
                                        simplex_edge_frame_message_scale_override=(
                                            simplex_edge_frame_message_scale_override
                                        ),
                                        simplex_boundary_edge_frame_gate_scale_override=(
                                            simplex_boundary_edge_frame_gate_scale_override
                                        ),
                                        simplex_boundary_readout_directionality_override=(
                                            simplex_boundary_readout_directionality_override
                                        ),
                                        simplex_boundary_hodge_readout_scale_override=(
                                            simplex_boundary_hodge_readout_scale_override
                                        ),
                                        simplex_boundary_edge_star_readout_scale_override=(
                                            simplex_boundary_edge_star_readout_scale_override
                                        ),
                                        simplex_boundary_edge_star_residual_scale_override=(
                                            simplex_boundary_edge_star_residual_scale_override
                                        ),
                                        simplex_vertex_star_context_scale_override=(
                                            simplex_vertex_star_context_scale_override
                                        ),
                                        simplex_edge_star_context_scale_override=(
                                            simplex_edge_star_context_scale_override
                                        ),
                                        simplex_pre_triangle_update_scale_override=(
                                            simplex_pre_triangle_update_scale_override
                                        ),
                                        simplex_pre_triangle_single_update_scale_override=(
                                            simplex_pre_triangle_single_update_scale_override
                                        ),
                                        simplex_segment_cell_scale_override=simplex_segment_cell_scale_override,
                                        simplex_msa_feedback_scale_override=simplex_msa_feedback_scale_override,
                                        simplex_boundary_pair_feedback_scale_override=(
                                            simplex_boundary_pair_feedback_scale_override
                                        ),
                                        simplex_boundary_pair_gate_scale_override=(
                                            simplex_boundary_pair_gate_scale_override
                                        ),
                                        simplex_boundary_metric_gate_scale_override=(
                                            simplex_boundary_metric_gate_scale_override
                                        ),
                                        simplex_triangle_attention_bias_scale_override=(
                                            simplex_triangle_attention_bias_scale_override
                                        ),
                                        simplex_triangle_attention_value_scale_override=(
                                            simplex_triangle_attention_value_scale_override
                                        ),
                                        simplex_local_neighbor_k_override=simplex_local_neighbor_k_override,
                                        simplex_geometry_distance_weight_override=(
                                            simplex_geometry_distance_weight_override
                                        ),
                                        simplex_face_top_k_override=simplex_face_top_k_override,
                                        simplex_tetra_top_k_override=simplex_tetra_top_k_override,
                                        simplex_cell_score_outer_edge_weight_override=(
                                            simplex_cell_score_outer_edge_weight_override
                                        ),
                                        use_reentrant=False,
                                    ),
                                )
                            else:
                                msa_repr, pair_repr, single_repr, simplex_aux = block(
                                    msa_repr,
                                    pair_repr,
                                    single_repr,
                                    msa_mask=evo_msa_mask,
                                    pair_mask=pair_mask,
                                    seq_mask=seq_mask,
                                    recycled_ca_coords=ca_prev,
                                    recycled_frames=rotations_prev,
                                    simplex_teacher_ca_coords=simplex_teacher_ca_coords,
                                    simplex_teacher_ca_mask=simplex_teacher_ca_mask,
                                    simplex_teacher_forcing_weight=simplex_teacher_forcing_weight,
                                    simplex_pair_update_scale_override=simplex_pair_update_scale_override,
                                    simplex_single_update_scale_override=simplex_single_update_scale_override,
                                    simplex_outer_edge_context_scale_override=(
                                        simplex_outer_edge_context_scale_override
                                    ),
                                    simplex_hodge_face_update_scale_override=(
                                        simplex_hodge_face_update_scale_override
                                    ),
                                    simplex_edge_frame_message_scale_override=(
                                        simplex_edge_frame_message_scale_override
                                    ),
                                    simplex_boundary_edge_frame_gate_scale_override=(
                                        simplex_boundary_edge_frame_gate_scale_override
                                    ),
                                    simplex_boundary_readout_directionality_override=(
                                        simplex_boundary_readout_directionality_override
                                    ),
                                    simplex_boundary_hodge_readout_scale_override=(
                                        simplex_boundary_hodge_readout_scale_override
                                    ),
                                    simplex_boundary_edge_star_readout_scale_override=(
                                        simplex_boundary_edge_star_readout_scale_override
                                    ),
                                    simplex_boundary_edge_star_residual_scale_override=(
                                        simplex_boundary_edge_star_residual_scale_override
                                    ),
                                    simplex_vertex_star_context_scale_override=(
                                        simplex_vertex_star_context_scale_override
                                    ),
                                    simplex_edge_star_context_scale_override=(
                                        simplex_edge_star_context_scale_override
                                    ),
                                    simplex_pre_triangle_update_scale_override=(
                                        simplex_pre_triangle_update_scale_override
                                    ),
                                    simplex_pre_triangle_single_update_scale_override=(
                                        simplex_pre_triangle_single_update_scale_override
                                    ),
                                    simplex_segment_cell_scale_override=simplex_segment_cell_scale_override,
                                    simplex_msa_feedback_scale_override=simplex_msa_feedback_scale_override,
                                    simplex_boundary_pair_feedback_scale_override=(
                                        simplex_boundary_pair_feedback_scale_override
                                    ),
                                    simplex_boundary_pair_gate_scale_override=(
                                        simplex_boundary_pair_gate_scale_override
                                    ),
                                    simplex_boundary_metric_gate_scale_override=(
                                        simplex_boundary_metric_gate_scale_override
                                    ),
                                    simplex_triangle_attention_bias_scale_override=(
                                        simplex_triangle_attention_bias_scale_override
                                    ),
                                    simplex_triangle_attention_value_scale_override=(
                                        simplex_triangle_attention_value_scale_override
                                    ),
                                    simplex_local_neighbor_k_override=simplex_local_neighbor_k_override,
                                    simplex_geometry_distance_weight_override=(
                                        simplex_geometry_distance_weight_override
                                    ),
                                    simplex_face_top_k_override=simplex_face_top_k_override,
                                    simplex_tetra_top_k_override=simplex_tetra_top_k_override,
                                    simplex_cell_score_outer_edge_weight_override=(
                                        simplex_cell_score_outer_edge_weight_override
                                    ),
                            )
                            if simplex_aux:
                                simplex_aux_last = simplex_aux
                                simplex_aux_current = simplex_aux
                        elif self.training:
                            msa_repr, pair_repr = cast(
                                tuple[torch.Tensor, torch.Tensor],
                                torch_checkpoint.checkpoint(
                                    block,
                                    msa_repr, pair_repr,
                                    msa_mask=evo_msa_mask, pair_mask=pair_mask,
                                    use_reentrant=False,
                                ),
                            )
                        else:
                            msa_repr, pair_repr = block(
                                msa_repr, pair_repr,
                                msa_mask=evo_msa_mask, pair_mask=pair_mask,
                            )
                            single_repr = self.single_rep_proj(msa_repr[:, 0, :, :])

                    # Algorithm 2 line 18: accumulate m_1i and z_ij only.
                    # Keep the last sample's full MSA rep (real-MSA rows only,
                    # dropping the appended template-angle rows) for the
                    # masked MSA head — see the comment above the accumulators.
                    single_rep_accum = single_rep_accum + msa_repr[:, 0, :, :]
                    structure_single_accum = structure_single_accum + single_repr
                    pair_repr_accum = pair_repr_accum + pair_repr
                    if (
                        (
                            self.simplex_structure_readout_scale > 0.0
                            or self.simplex_structure_pair_readout_scale > 0.0
                        )
                        and simplex_aux_current is not None
                    ):
                        if self.simplex_structure_readout_scale > 0.0:
                            simplex_single_readout = simplex_aux_current.get("simplex_structure_single_readout")
                            if simplex_single_readout is not None:
                                simplex_structure_single_accum = (
                                    simplex_structure_single_accum + simplex_single_readout
                                )
                        simplex_pair_readout = simplex_aux_current.get("simplex_structure_pair_readout")
                        if simplex_pair_readout is not None:
                            simplex_structure_pair_accum = simplex_structure_pair_accum + simplex_pair_readout
                    msa_repr_last = msa_repr[:, :msa_feat_current.shape[1], :, :]

                # Algorithm 2 line 20: m̂_1i, ẑ_ij /= N_ensemble.
                msa_first_row = single_rep_accum / n_ensemble
                pair_repr = pair_repr_accum / n_ensemble
                assert msa_repr_last is not None  # n_ensemble > 0 guarantees this
                msa_repr = msa_repr_last

                # Algorithm 6 line 12: s_i = Linear(m_1i). By linearity this is
                # equivalent to averaging s_i itself across ensemble members.
                if self.use_simplicial_evoformer:
                    single_rep = structure_single_accum / n_ensemble
                else:
                    single_rep = self.single_rep_proj(msa_first_row)
                if self.simplex_structure_readout_scale > 0.0:
                    readout_scale = self.simplex_structure_readout_scale
                    single_rep = single_rep + readout_scale * (simplex_structure_single_accum / n_ensemble)
                    pair_repr = pair_repr + readout_scale * (simplex_structure_pair_accum / n_ensemble)
                    single_rep = single_rep * seq_mask[..., None]
                    pair_repr = pair_repr * pair_mask[..., None]
                if self.simplex_structure_pair_readout_scale > 0.0:
                    pair_readout = _rms_normalize_last_dim(simplex_structure_pair_accum / n_ensemble)
                    pair_readout = pair_readout * pair_mask[..., None]
                    pair_repr = pair_repr + self.simplex_structure_pair_readout_scale * pair_readout
                    pair_repr = pair_repr * pair_mask[..., None]

                # Algorithm 2 line 21: StructureModule consumes (ŝ_i, ẑ_ij).
                structure_predictions = self.structure_model(
                    single_rep, pair_repr, aatype,
                    seq_mask=seq_mask, detach_rotations=detach_rotations,
                )

                if is_last:
                    # Auxiliary prediction heads (supplement 1.9). Each head's
                    # input is fixed by the paper:
                    #   - distogram (eq 41): averaged pair rep
                    #   - masked MSA (eq 42): last ensemble's full MSA rep
                    #   - experimentally resolved (1.9.10): Evoformer single
                    #     rep (s_i from Algorithm 6 line 12), NOT the SM single
                    #   - pLDDT (Algorithm 29 line 1): post-IPA single rep
                    #     from the Structure Module
                    #   - TM-score / PAE (1.9.7): averaged pair rep
                    distogram_logits = self.distogram_head(pair_repr)
                    masked_msa_logits = self.masked_msa_head(msa_repr)
                    experimentally_resolved_logits = self.experimentally_resolved_head(single_rep)
                    plddt_logits = self.plddt_head(structure_predictions["single"])
                    tm_logits = self.tm_score_head(pair_repr)

                    output = {
                        **structure_predictions,
                        "distogram_logits": distogram_logits,
                        "masked_msa_logits": masked_msa_logits,
                        "experimentally_resolved_logits": experimentally_resolved_logits,
                        "plddt_logits": plddt_logits,
                        "tm_logits": tm_logits,
                        "pair_representation": pair_repr,
                        "msa_representation": msa_repr,
                        "single_representation": single_rep,
                        "sampled_n_cycles": n_cycles,
                        "sampled_n_ensemble": n_ensemble,
                    }
                    if simplex_aux_last is not None:
                        output.update(
                            {
                                key: value
                                for key, value in simplex_aux_last.items()
                                if not key.startswith("simplex_structure_")
                            }
                        )
                    return output

                # Algorithm 2 line 22: store averaged m̂_1i, ẑ_ij (pre-projection,
                # in c_m not c_s) and pseudo-β positions for the next cycle.
                # detach() realises Algorithm 31 line 4 (no gradients between
                # iterations) — required since only the last cycle is unrolled
                # for backward.
                single_rep_prev = msa_first_row.detach()
                z_prev = pair_repr.detach()
                if simplex_boundary_metric_recycling_scale > 0.0 and simplex_aux_last is not None:
                    simplex_recycle_bins, simplex_recycle_mask = simplex_boundary_metric_recycling_bins(
                        simplex_aux_last,
                        num_residues=N_res,
                        n_recycle_bins=15,
                    )
                    simplex_recycle_bias = self.recycle_linear_d(simplex_recycle_bins.to(dtype=pair_repr.dtype))
                    simplex_recycle_bias = simplex_recycle_bias * simplex_recycle_mask.to(dtype=pair_repr.dtype)
                    simplex_recycle_bias = simplex_recycle_bias * pair_mask[..., None].to(dtype=pair_repr.dtype)
                    z_prev = z_prev + simplex_boundary_metric_recycling_scale * simplex_recycle_bias.detach()
                if simplex_boundary_cochain_recycling_scale > 0.0 and simplex_aux_last is not None:
                    simplex_pair_readout = simplex_aux_last.get("simplex_structure_pair_readout")
                    if simplex_pair_readout is not None:
                        simplex_cochain_bias = simplex_pair_readout.to(dtype=pair_repr.dtype)
                        cochain_metric_gate_scale = min(
                            max(self.simplex_boundary_cochain_recycling_metric_gate_scale, 0.0),
                            1.0,
                        )
                        if cochain_metric_gate_scale > 0.0:
                            simplex_confidence, simplex_confidence_mask = simplex_boundary_metric_confidence_map(
                                simplex_aux_last,
                                num_residues=N_res,
                            )
                            simplex_gate = 1.0 + cochain_metric_gate_scale * (2.0 * simplex_confidence - 1.0)
                            simplex_gate = torch.where(
                                simplex_confidence_mask > 0,
                                simplex_gate,
                                torch.ones_like(simplex_gate),
                            )
                            simplex_cochain_bias = simplex_cochain_bias * simplex_gate.to(dtype=pair_repr.dtype)
                        simplex_cochain_bias = simplex_cochain_bias * pair_mask[..., None].to(dtype=pair_repr.dtype)
                        z_prev = (
                            z_prev
                            + simplex_boundary_cochain_recycling_scale * simplex_cochain_bias.detach()
                        )

                # Pseudo-β: Cα for glycine (atom14 index 1, since GLY has no Cβ),
                # Cβ otherwise (atom14 index 4). Matches the pseudo-β convention
                # used throughout AF2 for pairwise distances (supplement 1.9.8,
                # Algorithm 32 line 1). aatype==7 is glycine in AF2's alphabet.
                is_gly = (aatype == 7)
                cb_idx = torch.where(is_gly, 1, 4)
                atom_coords = structure_predictions["atom14_coords"]
                ca_prev = atom_coords[:, :, 1, :].detach()
                rotations_prev = structure_predictions["final_rotations"].detach()
                x_prev = torch.gather(
                    atom_coords, 2,
                    cb_idx[:, :, None, None].expand(-1, -1, 1, 3),
                ).squeeze(2).detach()

        raise ValueError("n_cycles and n_ensemble must be > 0")
