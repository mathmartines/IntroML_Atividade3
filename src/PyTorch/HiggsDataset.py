import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class HiggsDataset(Dataset):
    """
    Loads the Higgs dataset with all the low level and high level features.

    Each event is described by a set of particles which each particle carries the following information:
    - particle pT, particle pseudo-rapidity (eta), particle azimuthal angle (phi), bTag,
      isLepton, isNeutrino, isJet

    Note that for neutrinos we only have the missing energy and the azimuthal angle phi, but we can infer
    the neutrino pseudo-rapidity by requiring the lepton-neutrino invariant mass equals the W boson mass.
    This procedure would be straingth foward if the data was not scaled, since we have this scaling we can not
    compare the lepton-neutrino invariant mass with the W-boson mass directly. But we do have one high-level
    feature, which is invariant mass of the lepton-neutrino system. The latter used a neutrino pseudo-rapidity value
    that lead to a lepton-neutrino mass close to the W boson mass. Instead of using the W mass, we use
    this high-level feature.
    When we reconstruct the neutrino pseudo-rapidity, we always have two possible solutions, we choose the one
    that leads to the smallest absolute value.

    We have other categorical features, which represents the following categories:
    - b-tagging: is only set to 1 for jets that are tagged as b-jets
    - isLepton: 1 for leptons 0 for other particles
    - isNeutrino: 1 for neutrinos (missing energy) 0 for other particles
    - isJet: 1 for jets 0 for other particles in the event.

    Here we assume that each event has the information in the following order
    - lepton pT, lepton eta, lepton phi, missing energy Et, missing energy phi,
      jet1 pT, jet1 eta, jet1 phi, jet1 bTag, j2 pT, j2 eta, j2 phi, j2 bTag, j3 pT,
      j3 eta, j3 phi, j3 bTag, j4 pT, j4 eta, j4 phi, j4 bTag
    """

    def __init__(self, file_path: str, device):
        self._higgs_dataset = pd.read_csv(file_path, header=None).to_numpy()
        self._particles_features = ["pt", "eta", "phi", "btag", "isLepton", "isNeutrino", "isJet"]
        self._device = device

    def __len__(self):
        return self._higgs_dataset.shape[0]

    def __getitem__(self, index: int):
        """Returns the set of particles in the data frame"""
        # the first column represens the label, the remaining one the features
        label_event, event_info = self._higgs_dataset[index][0], self._higgs_dataset[index][1:]
        # gathering the information about the particles
        lepton_features = torch.from_numpy(self._construct_lepton_features(event_info))
        neutrino_features = torch.from_numpy(self._construct_nu_features(event_info))
        jets_features = torch.from_numpy(self._construct_jets_features(event_info))
        event = torch.concat(
            [lepton_features, neutrino_features, jets_features],
            dim=0
        )
        return event.float().to(self._device), self._construct_label(label_event).float().to(self._device)

    @staticmethod
    def _construct_label(label_event):
        event_label = torch.tensor([int(index == label_event) for index in range(1, -1, -1)])
        return event_label

    def _construct_lepton_features(self, event_info: np.ndarray):
        """Constructs the lepton's features for the lepton in the event"""
        # the lepton momentum starts at index 0 and ends at index 2
        lepton_momentum = event_info[0: 3]
        # the only tag the lepton must have is the isLepton
        return np.array([self._initialize_particle_features(lepton_momentum, "isLepton")], dtype=np.float32)

    def _construct_jets_features(self, event_info: np.ndarray):
        """Constructs the features for each of the jets in the event"""
        number_jets = 4
        jets_features = np.empty(shape=(number_jets, len(self._particles_features)), dtype=np.float32)
        for jet_index in range(number_jets):
            # the first three indices of each jet represents the momentum
            jet_momentum = event_info[5 + 4 * jet_index: 5 + 4 * jet_index + 3]
            # the last index tells if the jet is from a b or not
            tags = ["isJet", "btag"] if event_info[5 + 4 * jet_index + 3] > 1 else ["isJet"]
            jets_features[jet_index] = self._initialize_particle_features(jet_momentum, *tags)
        return jets_features

    def _construct_nu_features(self, event_info: np.ndarray):
        """
        Constructs the neutrino features for the event.
        For the neutrino pseudo-rapidity it uses the procedures described in the description of the
        class.
        """
        # lepton_pt, lepton_eta, lepton_phi = event_info[:3]
        missing_energy, missing_energy_phi = event_info[3:5]
        # neutrino momentum
        nu_momentum = np.array([missing_energy, 0, missing_energy_phi], dtype=np.float32)
        return np.array([self._initialize_particle_features(nu_momentum, "isNeutrino")])

    def _initialize_particle_features(self, momentum, *tags):
        particle_features = np.zeros(len(self._particles_features))
        # adding the momentum to the features
        particle_features[0: 3] = momentum
        # adding all the tags for the particle
        mask_indices = [self._particles_features.index(particle_tag) for particle_tag in tags]
        particle_features[mask_indices] = 1
        return particle_features




