from math import sqrt

import numpy as np
from hlt.entity import Ship, Position
from . import collision, entity
from .constants import DIST_POWER


class Map:
    """
    Map which houses the current game information/metadata.

    :ivar my_id: Current player id associated with the map
    :ivar width: Map width
    :ivar height: Map height
    """

    def __init__(self, my_id, width, height):
        """
        :param my_id: User's id (tag)
        :param width: Map width
        :param height: Map height
        """
        self.my_id = my_id
        self.width = width
        self.height = height
        self._players = {}
        self._planets = {}
        self.gridSet = None
        self.params = None

    def set_gridSet(self, gridSet):
        self.gridSet = gridSet

    def get_me(self):
        """
        :return: The user's player
        :rtype: Player
        """
        return self._players.get(self.my_id)

    def get_player(self, player_id):
        """
        :param int player_id: The id of the desired player
        :return: The player associated with player_id
        :rtype: Player
        """
        return self._players.get(player_id)

    def all_players(self):
        """
        :return: List of all players
        :rtype: list[Player]
        """
        return list(self._players.values())

    def get_planet(self, planet_id):
        """
        :param int planet_id:
        :return: The planet associated with planet_id
        :rtype: entity.Planet
        """
        return self._planets.get(planet_id)

    @property
    def all_entities(self):
        """
        Handle for all entities
        :return: All entities
        :rtype: [Entity]
        """
        return self._all_ships() + self.all_planets()

    def all_planets(self):
        """
        :return: List of all planets
        :rtype: list[entity.Planet]
        """
        return list(self._planets.values())

    def augment_entities(self, params):
        """
        Add info about ME player on all entities and the keyword arguments
        used to run the script
        :param params:
        :type params:
        :return:
        :rtype:
        """
        self.params = params
        for e in self._all_ships() + self.all_planets():
            e.augment(self.get_me(), params)

    def get_nearest_planet(self, entity):
        """

        :param entity: The source entity to find distances from
        :type entity: Entity
        :return: Nearest Planet
        :rtype: Planet
        """
        return min(self.all_planets(),
                   key=lambda p: (p.x - entity.x) ** 2 + (p.y - entity.y) ** 2)

    def nearby_entities_by_distance(self, entity):
        """
        :param entity: The source entity to find distances from
        :return: Dict containing all entities with their designated distances
        :rtype: dict
        """
        result = {}
        for foreign_entity in self._all_ships() + self.all_planets():
            if entity == foreign_entity:
                continue
            result.setdefault(entity.calculate_distance_between(foreign_entity),
                              []).append(foreign_entity)
        return result

    def _link(self):
        """
        Updates all the entities with the correct ship and planet objects

        :return:
        """
        for celestial_object in self.all_planets() + self._all_ships():
            celestial_object._link(self._players, self._planets)

    def _parse(self, map_string):
        """
        Parse the map description from the game.

        :param map_string: The string which the Halite engine outputs
        :return: nothing
        """
        tokens = map_string.split()

        self._players, tokens = Player._parse(tokens)
        self._planets, tokens = entity.Planet._parse(tokens)

        assert (
                len(
                        tokens) == 0)  # There should be no remaining tokens
        # at this
        # point
        self._link()

    def _all_ships(self):
        """
        Helper function to extract all ships from all players

        :return: List of ships
        :rtype: List[Ship]
        """
        all_ships = []
        for player in self.all_players():
            all_ships.extend(player.all_ships())
        return all_ships

    def all_ennemy_ships(self):
        """
        Helper function to extract all ships from all players but me

        :return: List of ships
        :rtype: List[Ship]
        """
        all_ships = []
        for player in self.all_players():
            if player != self.get_me():
                all_ships.extend(player.all_ships())
        return all_ships

    def compute_planet_threat_attractivity(self):
        """
        Compute a threat level for the vicinity of a planet
        Helps to decide wether or not to dock/undock

        """
        if self.gridSet is None:
            raise AttributeError
        for planet in self.all_planets():
            entities = self.gridSet.get_neighbours(planet, lvl=2)
            # list of ones for my ship, zeros for his
            ships = [int(entity.isMine) for entity in entities if
                     entity.isMobile]
            tot_ships = len(ships)
            foe_presence = tot_ships - sum(ships)
            n_docked = len(planet.all_docked_ships)
            f = self.params
            if planet.owner is None:
                attractivity = f["explore"]*planet.free_spots * (
                        1 + 1 / planet.num_docking_spots) - tot_ships
            elif planet.owner == self.get_me():
                attractivity = f["defend"]*(foe_presence - sum(ships)) * sqrt(
                        n_docked)
            else:
                attractivity = f["raid"]*n_docked

            threat_level = foe_presence / max(tot_ships - 1, 1)

            planet.set_threat(threat_level)
            planet.set_attractivity(attractivity)

    def assign_targets(self):
        """
        Assign a target for each ship using planet attractivity
        The attractivity is discounted by the distance
        """
        my_ships = self.my_undocked_ships
        planets = self.all_planets()
        ship_pos = np.array([[e.x, e.y] for e in my_ships])
        planet_pos = np.array([[e.x, e.y] for e in planets])
        planet_att = np.array([e.attractivity_level
                               for e in planets])
        n_ship = ship_pos.shape[0]
        n_planet = planet_pos.shape[0]
        diff_pos = (np.broadcast_to(ship_pos.reshape(n_ship, 1, 2),
                                    (n_ship, n_planet, 2)) -
                    np.broadcast_to(planet_pos.reshape(1, n_planet, 2),
                                    (n_ship, n_planet, 2)))
        dist = np.sum(diff_pos * diff_pos, axis=2)
        dist = np.power(dist, DIST_POWER)
        attractivity = dist * planet_att
        for i, ship in enumerate(my_ships):
            planet_ind = np.argmax(attractivity[i, :])
            ship.target = Position.from_entity(planets[planet_ind])

    @property
    def my_undocked_ships(self):
        """
        Shortcut to retrieve only my undocked ships
        :return:
        :rtype:
        """
        return [ship for ship in self.get_me().all_ships()
                if ship.docking_status == Ship.DockingStatus.UNDOCKED]

    @property
    def my_docked_ships(self):
        """
        Shortcut to retrieve only my docked ships
        :return:
        :rtype:
        """
        return [ship for ship in self.get_me().all_ships()
                if ship.docking_status != Ship.DockingStatus.DOCKED]

    def _intersects_entity(self, target):
        """
        Check if the specified entity (x, y, r) intersects any planets.
        Entity is assumed to not be a planet.

        :param entity.Entity target: The entity to check intersections with.
        :return: The colliding entity if so, else None.
        :rtype: entity.Entity
        """
        for celestial_object in self._all_ships() + self.all_planets():
            if celestial_object is target:
                continue
            d = celestial_object.calculate_distance_between(target)
            if d <= celestial_object.radius + target.radius + 0.1:
                return celestial_object
        return None

    def obstacles_between(self, ship, target, ignore=()):
        """
        Check whether there is a straight-line path to the given point,
        without planetary obstacles in between.

        :param entity.Ship ship: Source entity
        :param entity.Entity target: Target entity
        :param entity.Entity ignore: Which entity type to ignore
        :return: The list of obstacles between the ship and target
        :rtype: list[entity.Entity]
        """
        obstacles = []
        entities = ([] if issubclass(entity.Planet,
                                     ignore) else self.all_planets()) \
                   + (
                       [] if issubclass(entity.Ship,
                                        ignore) else self._all_ships())
        for foreign_entity in entities:
            if foreign_entity == ship or foreign_entity == target:
                continue
            if collision.intersect_segment_circle(ship, target, foreign_entity,
                                                  fudge=ship.radius + 0.1):
                obstacles.append(foreign_entity)
        return obstacles


class Player:
    """
    :ivar id: The player's unique id
    """

    def __init__(self, player_id, ships={}):
        """
        :param player_id: User's id
        :param ships: Ships user controls (optional)
        """
        self.id = player_id
        self._ships = ships

    def all_ships(self):
        """
        :return: A list of all ships which belong to the user
        :rtype: list[entity.Ship]
        """
        return list(self._ships.values())

    def get_ship(self, ship_id):
        """
        :param int ship_id: The ship id of the desired ship.
        :return: The ship designated by ship_id belonging to this user.
        :rtype: entity.Ship
        """
        return self._ships.get(ship_id)

    @staticmethod
    def _parse_single(tokens):
        """
        Parse one user given an input string from the Halite engine.

        :param list[str] tokens: The input string as a list of str from the
        Halite engine.
        :return: The parsed player id, player object, and remaining tokens
        :rtype: (int, Player, list[str])
        """
        player_id, *remainder = tokens
        player_id = int(player_id)
        ships, remainder = entity.Ship._parse(player_id, remainder)
        player = Player(player_id, ships)
        return player_id, player, remainder

    @staticmethod
    def _parse(tokens):
        """
        Parse an entire user input string from the Halite engine for all users.

        :param list[str] tokens: The input string as a list of str from the
        Halite engine.
        :return: The parsed players in the form of player dict, and remaining
        tokens
        :rtype: (dict, list[str])
        """
        num_players, *remainder = tokens
        num_players = int(num_players)
        players = {}

        for _ in range(num_players):
            player, players[player], remainder = Player._parse_single(remainder)

        return players, remainder

    def __str__(self):
        return "Player {} with ships {}".format(self.id, self.all_ships())

    def __repr__(self):
        return self.__str__()
