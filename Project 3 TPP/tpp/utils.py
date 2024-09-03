from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
def get_sequence_batch(
    inter_event_times: List[TensorType[torch.float32]],
) -> Tuple[
    TensorType[torch.float32, "batch", "max_seq_length"],
    TensorType[torch.bool, "batch", "max_seq_length"],
]:
    """
    Generate padded batch and mask for list of sequences.

        Args:
            inter_event_times (List): list of inter-event times

        Returns:
            batch: batched inter-event times. shape [batch_size, max_seq_length]
            mask: boolean mask indicating inter-event times. shape [batch_size, max_seq_length]
    """

    #######################################################
    # write here
    max_seq_length = max(len(seq) for seq in inter_event_times)
    batch_size = len(inter_event_times)

    batch = torch.zeros((batch_size, max_seq_length), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_seq_length), dtype=torch.bool)

    for i, seq in enumerate(inter_event_times):
        seq_len = len(seq)
        batch[i, :seq_len] = seq
        mask[i, :seq_len] = 1
    #######################################################

    return batch, mask


@typechecked
def get_tau(
    t: TensorType[torch.float32, "sequence_length"], t_end: TensorType[torch.float32, 1]
) -> TensorType[torch.float32]:
    """
    Compute inter-eventtimes from arrival times

        Args:
            t: arrival times. shape [seq_length]
            t_end: end time of the temporal point process.

        Returns:
            tau: inter-eventtimes.
    """
    # compute inter-eventtimes
    #######################################################
    # write here
    tau = torch.zeros(t.shape[0] + 1)
    tau[1:-1] = t[1:] - t[:-1]
    tau[0] = t[0]
    tau[-1] = t_end - t[-1]
    #######################################################

    return tau
