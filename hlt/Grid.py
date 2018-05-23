from hlt.entity import Centroid
import numpy as np
from math import exp


class Grid(object):
    """
    2D-Grid representation of the game map. Contains a directory of all
    entities address. Support the computation of centroids which are entities
    whose energy is a summary of the entities registered in one space.

    """

    def __init__(self, grid_size):
        self.size = grid_size
        self.ledger = {}
        self.centroids = {}

    def entity2add(self, entity):
        row = entity.x // self.size
        col = entity.y // self.size
        return int(row), int(col)

    def add2coo(self, row, col):
        x = (row + 0.5) * self.size
        y = (col + 0.5) * self.size
        return x, y

    def _register_entity(self, entity):
        """
        Add the entity to its address in the ledger
        :param entity: Entity to register
        :type entity: Entity
        """
        address = self.entity2add(entity)

        row = self.ledger.setdefault(address[0], {})
        row.setdefault(address[1], []).append(entity)

    def register_entities(self, entities):
        """
        Register the entities to the Grid object. They are then accessible
        through the ledger
        :param entities: list of entities to be registered
        :type entities: [Entity]
        """
        for entity in entities:
            self._register_entity(entity)

    def _ant_process(self, entities):
        """

        :param entities:
        :type entities:
        :return:
        :rtype:
        """
        all_factors = np.array([e.factors for e in entities])
        all_kernels = np.array([e.kernels for e in entities])
        kernel_ref = np.max(all_kernels, axis=0)

        factors = np.sum(all_factors *
                         np.exp((self.size / kernel_ref.reshape(1, 2)) ** 2 -
                                (self.size / all_kernels) ** 2),
                         axis=0)

        kernels = (kernel_ref[0], kernel_ref[1])
        factors = (factors[0], factors[1])

        return factors, kernels

    def _compute_centroid(self, row, col, entities):
        """
        Add a Centroid entity to its place on the Grid. The factor and kernel
        associated with this Centroid entity are computed with the ant
        process method.
        :param row: row index
        :type row: int
        :param col: column index
        :type col: int
        :param entities: list of entities registerd inside this space
        :type entities: [Entity]
        """

        x, y = self.add2coo(row, col)
        factors, kernels = self._ant_process(entities)
        self.centroids[row][col] = Centroid(x, y, factors, kernels)

    def compute_centroids(self):
        """
        Add a Centroid entity to each space containing at least one entity. The
        factor and kernel associated with this Centroid entity are computed
        with the ant process method.
        :return:
        :rtype:
        """
        for row, columns in self.ledger.items():
            self.centroids[row] = {}
            for col, entities in columns.items():
                self._compute_centroid(row, col, entities)

    def get_neighbours(self, entity, dist=1):
        """

        :param entity:
        :type entity:
        :param dist: set to 0 to get all entities in the same space, set to 1
        to get all entities in a 3x3 grid
        :type dist: int
        :return: All the entities registered at the same address as the
        reference AND all entities registered at max
        :rtype: [Entity]
        """
        row, col = self.entity2add(entity)
        list_row = range(row - dist, row + 1 + dist)
        list_col = range(col - dist, col + 1 + dist)
        list_entities = []
        for x in list_row:
            for y in list_col:
                # if dist == 1 and (x, y) == (row, col):
                #     continue
                list_entities.extend(self.ledger.get(x, {}).get(y, []))

        return list_entities

    def get_centroids(self, entity, dist=1):
        """
        Get all the centroids at dist of the reference
        :param entity:
        :type entity:
        :param dist: set to 0 return None, set to 1  to get all entities in a
        3x3 grid except the central one
        :type dist: int
        :return: All the centroids at dist of the reference
        :rtype: [Centroid]
        """

        row, col = self.entity2add(entity)
        list_row = range(row - dist, row + 1 + dist)
        list_col = range(col - dist, col + 1 + dist)
        centroids = []
        for x in list_row:
            for y in list_col:
                if dist == 1 and (x, y) == (row, col):
                    continue
                centroid = self.centroids.get(x, {}).get(y, None)
                if centroid is not None:
                    centroids.append(centroid)

        return centroids


class GridSet(object):

    def __init__(self, sizes, entities):
        self.sizes = sizes
        self.grids = [Grid(size) for size in sizes]
        self.register_entities(entities)
        self.compute_centroids()

    def register_entities(self, entities):
        """
        Register the entities to the Grid objects. They are then accessible
        through the ledger
        :param entities: list of entities to be registered
        :type entities: [Entity]
        """
        for grid in self.grids:
            grid.register_entities(entities)

    def compute_centroids(self, ):
        """
        Add a Centroid entity to each space containing at least one entity. The
        factor and kernel associated with this Centroid entity are computed
        with the ant process method.
        :return:
        :rtype:
        """
        for grid in self.grids:
            grid.compute_centroids()

    def get_neighbours(self, entity, lvl=0, **kwargs):
        """

        :param entity:
        :type entity:
        :param lvl: lvl at which you want to get neighbours
        :type lvl:
        :return: All the entities registered at the same address as the
        reference AND all entities registered at max
        :rtype: [Entity]
        """
        return self.grids[lvl].get_neighbours(entity, **kwargs)

    def get_centroids(self, entity, skip_first=True, **kwargs):
        """
        Get all the centroids at dist of the reference
        :param entity:
        :type entity:
        :param skip_first: avoid getting centroid for first level
        :type skip_first: bool
        :param dist: set to 0 return None, set to 1  to get all entities in a
        3x3 grid except the central one
        :type dist: int
        :return: All the centroids at dist of the reference
        :rtype: [Centroid]
        """
        list_centroids = []
        for grid in self.grids[int(skip_first):]:
            list_centroids.extend(grid.get_centroids(entity, **kwargs))
        return list_centroids
