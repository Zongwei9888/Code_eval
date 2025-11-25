import numpy as np

class GraphInversion:
    """
    Implements the Graph Inversion algorithm as described in the paper's addendum.
    This algorithm computes a dynamic attention mask (H) for the Transformer based on
    the base dependency graph (M_E) and the set of latent variables (M_C).
    """
    def __init__(self, m_e: np.ndarray):
        """
        Initializes the GraphInversion algorithm with the base dependency graph.

        Args:
            m_e (np.ndarray): The base dependency graph (M_E) as a boolean adjacency matrix.
        """
        self.g = m_e.astype(bool)
        self.num_vars = self.g.shape[0]

    def _moralize(self, g: np.ndarray) -> np.ndarray:
        """
        Moralizes the graph G.
        1. Makes the graph undirected.
        2. Connects all parents of each node.

        Args:
            g (np.ndarray): The input graph as an adjacency matrix.

        Returns:
            np.ndarray: The moralized, undirected graph J.
        """
        j = g | g.T
        for i in range(self.num_vars):
            parents = np.where(g[:, i])[0]
            if len(parents) > 1:
                for p1_idx in range(len(parents)):
                    for p2_idx in range(p1_idx + 1, len(parents)):
                        p1 = parents[p1_idx]
                        p2 = parents[p2_idx]
                        j[p1, p2] = True
                        j[p2, p1] = True
        return j

    def _min_fill_criterion(self, v: int, j: np.ndarray, unmarked_neighbors: np.ndarray) -> int:
        """
        Calculates the number of edges that would be added to J if v were selected.
        This is the "fill-in" cost.

        Args:
            v (int): The vertex being considered.
            j (np.ndarray): The current moralized graph.
            unmarked_neighbors (np.ndarray): An array of unmarked neighbors of v.

        Returns:
            int: The number of edges that would need to be added.
        """
        fill_in_count = 0
        for i in range(len(unmarked_neighbors)):
            for k in range(i + 1, len(unmarked_neighbors)):
                n1 = unmarked_neighbors[i]
                n2 = unmarked_neighbors[k]
                if not j[n1, n2]:
                    fill_in_count += 1
        return fill_in_count

    def compute_attention_mask(self, m_c: np.ndarray) -> np.ndarray:
        """
        Computes the final attention mask H based on the latent variables.

        Args:
            m_c (np.ndarray): A boolean mask indicating the latent variables (Z).

        Returns:
            np.ndarray: The computed attention mask H.
        """
        # 1. J <- MORALIZE(G)
        j = self._moralize(self.g)

        # 2. Set all vertices of J to be unmarked.
        marked = np.zeros(self.num_vars, dtype=bool)

        # 3. H <- Graph with vertices of G and no edges.
        h = np.zeros_like(self.g, dtype=bool)

        # 4. S <- Set of all latent variables in Z that have no latent parents in G.
        s = []
        latent_indices = np.where(m_c)[0]
        for v in latent_indices:
            parents = np.where(self.g[:, v])[0]
            latent_parents = [p for p in parents if m_c[p]]
            if not latent_parents:
                s.append(v)

        # 5. while S is not empty do:
        while s:
            # 6. Select v from S according to a min-fill criterion.
            min_fill = float('inf')
            v_selected = -1
            for v_candidate in s:
                unmarked_neighbors = np.where(j[v_candidate] & ~marked)[0]
                fill = self._min_fill_criterion(v_candidate, j, unmarked_neighbors)
                if fill < min_fill:
                    min_fill = fill
                    v_selected = v_candidate
            
            v = v_selected
            s.remove(v)

            unmarked_neighbors_of_v = np.where(j[v] & ~marked)[0]

            # 7. Add edges in J between all unmarked neighbors of v.
            for i in range(len(unmarked_neighbors_of_v)):
                for k in range(i + 1, len(unmarked_neighbors_of_v)):
                    n1 = unmarked_neighbors_of_v[i]
                    n2 = unmarked_neighbors_of_v[k]
                    j[n1, n2] = True
                    j[n2, n1] = True

            # 8. Make unmarked neighbors of v in J become parents of v in H.
            for neighbor in unmarked_neighbors_of_v:
                h[neighbor, v] = True

            # 9. Mark v and remove it from S.
            marked[v] = True

            # 10. For each unmarked child latent u of v in G:
            children = np.where(self.g[v, :])[0]
            for u in children:
                if m_c[u] and not marked[u]:
                    # 11. If all parent latents of u in G are marked, add u to S.
                    parents_of_u = np.where(self.g[:, u])[0]
                    latent_parents_of_u = [p for p in parents_of_u if m_c[p]]
                    if all(marked[p] for p in latent_parents_of_u):
                        s.append(u)

        # 14. return H.
        return h
