from dataclasses import dataclass, fields
from jaxtyping import Array, Float, Long

def flatten_dataclass(cls: type):
    """Return a function that flattens a dataclass into a list"""
    cls_fields = [f.name for f in fields(cls)]
    return lambda x: [getattr(x, f) for f in cls_fields]


def unflatten_dataclass(cls: type):
    """Return a function that unflattens a list into a dataclass"""
    cls_fields = [f.name for f in fields(cls)]
    return lambda x: cls(**dict(zip(cls_fields, x)))


@dataclass
class ILQLElement:
    """
    Data element for ILQL

    :param input_ids: Input tokens. Should be a long tensor.
    :type input_ids: jax.numpy.ndarray

    :param attention_mask: Attention mask. Should be a long tensor.
    :type attention_mask: jax.numpy.ndarray

    :param rewards: Rewards for each token. Should be a float tensor of same size as tokens.
    :type rewards: jax.numpy.ndarray
    """
    input_ids: Long[Array, "query_size"]
    attention_mask: Long[Array, "query_size"]
    rewards: Float[Array, "reward_size"]
    states_ixs: Float[Array, "states_size"]
    actions_ixs: Float[Array, "reward_size"]
    dones: Float[Array, "states_size"]


@dataclass
class ILQLSeq2SeqElement:
    """
    Data element for ILQL

    :param input_ids: Input tokens. Should be a long tensor.
    :type input_ids: jax.numpy.ndarray

    :param attention_mask: Attention mask. Should be a long tensor.
    :type attention_mask: jax.numpy.ndarray

    :param rewards: Rewards for each token. Should be a float tensor of same size as tokens.
    :type rewards: jax.numpy.ndarray
    """
    input_ids: Long[Array, "query_size"]
    attention_mask: Long[Array, "query_size"]
    decoder_input_ids: Float[Array, "reward_size"]
    rewards: Float[Array, "reward_size"]
    states_ixs: Float[Array, "states_size"]
    actions_ixs: Float[Array, "reward_size"]
    dones: Float[Array, "states_size"]


@dataclass
class ILQLBatch:
    """
    Batched ILQL data elements

    :param input_ids: Batch of input tokens.
    :type input_ids: jax.numpy.ndarray

    :param attention_mask: Batch of attention masks.
    :type attention_mask: jax.numpy.ndarray

    :param rewards: Batch of rewards for each token in each token batch.
    :type rewards: jax.numpy.ndarray
    """

    input_ids: Float[Array, "batch_size", "query_size"]
    attention_mask: Float[Array, "batch_size", "query_size"]
    rewards: Float[Array, "batch_size", "reward_size"]
    states_ixs: Float[Array, "batch_size", "states_size"]
    actions_ixs: Float[Array, "batch_size", "reward_size"]
    dones: Float[Array, "batch_size", "states_size"]


@dataclass
class ILQLSeq2SeqBatch:
    """
    Batched ILQL data elements

    :param input_ids: Batch of input tokens.
    :type input_ids: jax.numpy.ndarray

    :param attention_mask: Batch of attention masks.
    :type attention_mask: jax.numpy.ndarray

    :param rewards: Batch of rewards for each token in each token batch.
    :type rewards: jax.numpy.ndarray
    """
    input_ids: Float[Array, "batch_size", "query_size"]
    attention_mask: Float[Array, "batch_size", "query_size"]
    decoder_input_ids: Float[Array, "batch_size", "reward_size"]
    rewards: Float[Array, "batch_size", "reward_size"]
    states_ixs: Float[Array, "batch_size", "states_size"]
    actions_ixs: Float[Array, "batch_size", "reward_size"]
    dones: Float[Array, "batch_size", "states_size"]
