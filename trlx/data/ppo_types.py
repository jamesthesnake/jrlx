from dataclasses import dataclass
from jaxtyping import Array, Float, Long


@dataclass
class PPORLElement:
    """
    :param query_tensor: The query tensor i.e. the prompt tokens.
                         Should be a long tensor.
    :type query_tensor: jax.numpy.ndarray

    :param response_tensor: The response tensor i.e. the output tokens.
                            Should be a long tensor.
    :type response_tensor: jax.numpy.ndarray

    :param logprobs: The log probabilities over the response tokens generated
                    by the policy network (i.e. the autoregressive model).
                    Should be a float tensor of same size as tokens.
    :type logprobs: jax.numpy.ndarray

    :param values: The values for each token generated from the value network or value head.
                    Should be a float tensor of same size as tokens.
    :type values: jax.numpy.ndarray

    :param rewards: The rewards for each token outputted in response.
                    Should be a float tensor of same size as tokens.
    :type rewards: jax.numpy.ndarray
    """
    query_tensor: Long[Array, "query_size"]
    response_tensor: Long[Array, "response_size"]
    logprobs: Float[Array, "response_size"]
    values: Float[Array, "response_size"]
    rewards: Float[Array, "response_size"]


@dataclass
class PPORLBatch:
    """
    A batched version of the PPORLElement. See PPORLElement for more details on individual fields.

    :param query_tensors: A batch of query tensors. Should be a long tensor.
    :type query_tensors: jax.numpy.ndarray

    :param response_tensors: A batch of response tensors. Should be a long tensor.
    :type response_tensors: jax.numpy.ndarray

    :param logprobs: A batch of log probabilities from policy
    :type logprobs: jax.numpy.ndarray

    :param values: A batch of values from value network
    :type values: jax.numpy.ndarray

    :param rewards: A batch of rewards
    :type rewards: jax.numpy.ndarray
    """
    query_tensors: Long[Array, "batch_size", "query_size"]
    response_tensors: Long[Array, "batch_size", "response_size"]
    logprobs: Float[Array, "batch_size", "response_size"]
    values: Float[Array, "batch_size", "response_size"]
    rewards: Float[Array, "batch_size", "response_size"]
