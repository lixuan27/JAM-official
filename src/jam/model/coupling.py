"""Attention masks that define the coupling between the world and action streams."""

from __future__ import annotations

from dataclasses import dataclass

import torch

VISIBILITY = {
    "reciprocal": (True, True),
    "world_to_action": (True, False),
    "action_to_world": (False, True),
    "decoupled": (False, False),
}


@dataclass(frozen=True)
class CouplingSpec:
    visibility: str = "reciprocal"

    def __post_init__(self):
        if self.visibility not in VISIBILITY:
            raise ValueError(f"unknown coupling visibility {self.visibility!r}")

    @property
    def action_sees_world(self) -> bool:
        return VISIBILITY[self.visibility][0]

    @property
    def world_sees_action(self) -> bool:
        return VISIBILITY[self.visibility][1]


@dataclass
class TokenBook:
    world_tokens: int
    clean_world_tokens: int
    agent_condition_tokens: int
    action_tokens: int
    consequence_tokens: int = 0

    @property
    def agent_tokens(self) -> int:
        return self.agent_condition_tokens + self.action_tokens + self.consequence_tokens

    @property
    def total(self) -> int:
        return self.world_tokens + self.agent_tokens

    @property
    def world(self) -> slice:
        return slice(0, self.world_tokens)

    @property
    def clean(self) -> slice:
        return slice(0, self.clean_world_tokens)

    @property
    def noisy_world(self) -> slice:
        return slice(self.clean_world_tokens, self.world_tokens)

    @property
    def agent(self) -> slice:
        return slice(self.world_tokens, self.total)

    @property
    def condition(self) -> slice:
        return slice(self.world_tokens, self.world_tokens + self.agent_condition_tokens)

    @property
    def action(self) -> slice:
        s = self.world_tokens + self.agent_condition_tokens
        return slice(s, s + self.action_tokens)

    @property
    def consequence(self) -> slice:
        s = self.world_tokens + self.agent_condition_tokens + self.action_tokens
        return slice(s, s + self.consequence_tokens)


def build_joint_mask(spec: CouplingSpec, book: TokenBook, device=None) -> torch.Tensor:
    m = torch.zeros(book.total, book.total, dtype=torch.bool, device=device)
    W, NW, K = book.world, book.noisy_world, book.clean
    AG, CND, ACT, CSQ = book.agent, book.condition, book.action, book.consequence
    reciprocal = spec.visibility == "reciprocal"

    m[W, W] = True
    m[K, :] = False
    m[K, K] = True

    m[AG, AG] = True
    m[AG, K] = True
    if not reciprocal:
        m[CND, :] = False
        m[CND, CND] = True
        m[CND, K] = True
    m[NW, CND] = True

    if spec.action_sees_world:
        m[ACT, W] = True
        if reciprocal:
            m[CND, W] = True
        if book.consequence_tokens:
            m[CSQ, W] = True
    if spec.world_sees_action:
        m[NW, ACT] = True
        if book.consequence_tokens:
            m[NW, CSQ] = True
    return m
