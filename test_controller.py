# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

class TestController(KesslerController):
    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        self.eval_frames = 0

        # Fuzzy variables
        # Antecedent
        bullet_time = ctrl.Antecedent(np.arange(0,1.01,0.01), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi, math.pi, 0.1), 'theta_delta')
        asteroid_dist = ctrl.Antecedent(np.arange(0, 800, 1), 'asteroid_dist')
        ship_thrust_a = ctrl.Antecedent(np.arange(-480, 480, 1), 'ship_thrust_a')
        
        # Consequents
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')
        ship_thrust_c = ctrl.Consequent(np.arange(-480, 480, 1), 'ship_thrust_c')
        ship_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_mine')

        # Membership functions
        # Bullet time
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.5])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, 0.5, 1.0])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.5, 1.0)

        # Theta delta
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3, -1*math.pi/6)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/2, -1*math.pi/4, 0])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/6, -1*math.pi/12, math.pi/30])
        theta_delta['Z']  = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, 0, math.pi/30])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-math.pi/30, math.pi/12, math.pi/6])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [0, math.pi/4, math.pi/2])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, math.pi/6, math.pi/3)

        # Asteroid distance
        asteroid_dist['Close']  = fuzz.zmf(asteroid_dist.universe, 0, 200)
        asteroid_dist['Medium'] = fuzz.trimf(asteroid_dist.universe, [150, 400, 650])
        asteroid_dist['Far']    = fuzz.smf(asteroid_dist.universe, 600, 800)

        # Ship turn
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -90])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-135, -90, -45])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90, -45, 0])
        ship_turn['Z']  = fuzz.trimf(ship_turn.universe, [-10, 0, 10])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0, 45, 90])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [45, 90, 135])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [90, 180, 180])

        # Ship fire
        ship_fire['No']  = fuzz.trimf(ship_fire.universe, [-1, -1, 0])
        ship_fire['Yes'] = fuzz.trimf(ship_fire.universe, [0, 1, 1])

        # Ship thrust
        ship_thrust['Back'] = fuzz.trimf(ship_thrust.universe, [-480, -480, 0])
        ship_thrust['Stop'] = fuzz.trimf(ship_thrust.universe, [-10, 0, 10])
        ship_thrust['Fwd']  = fuzz.trimf(ship_thrust.universe, [0, 480, 480])

        # Ship mine
        ship_mine['No']  = fuzz.trimf(ship_mine.universe, [-1, -1, 0])
        ship_mine['Yes'] = fuzz.trimf(ship_mine.universe, [0, 1, 1])

        # Rules
        # Steering rules
        rule1 = ctrl.Rule(theta_delta['NL'], ship_turn['NL'])
        rule2 = ctrl.Rule(theta_delta['NM'], ship_turn['NM'])
        rule3 = ctrl.Rule(theta_delta['NS'], ship_turn['NS'])
        rule4 = ctrl.Rule(theta_delta['Z'], ship_turn['Z'])
        rule5 = ctrl.Rule(theta_delta['PS'], ship_turn['PS'])
        rule6 = ctrl.Rule(theta_delta['PM'], ship_turn['PM'])
        rule7 = ctrl.Rule(theta_delta['PL'], ship_turn['PL'])
        # Firing rules
        rule8 = ctrl.Rule(bullet_time['S'], ship_fire['Yes'])
        rule9 = ctrl.Rule(bullet_time['M'] & (asteroid_dist['Close'] | asteroid_dist['Medium']), ship_fire['Yes'])
        rule10 = ctrl.Rule(bullet_time['L'] & asteroid_dist['Far'], ship_fire['No'])
        # Thrust rules
        rule11 = ctrl.Rule(asteroid_dist['Close'], ship_thrust['Stop'])
        rule12 = ctrl.Rule(asteroid_dist['Medium'], ship_thrust['Stop'])
        rule13 = ctrl.Rule(asteroid_dist['Far'], ship_thrust['Fwd'])
        # Mine rules
        rule14 = ctrl.Rule(asteroid_dist['Close'] & bullet_time['S'], ship_mine['Yes'])
        rule15 = ctrl.Rule(asteroid_dist['Medium'] & bullet_time['S'], ship_mine['No'])
        rule16 = ctrl.Rule(asteroid_dist['Far'] | bullet_time['L'] | bullet_time['L'], ship_mine['No'])

        # Control system
        self.control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7,
                                                  rule8, rule9, rule10,
                                                  rule11, rule12, rule13,
                                                  rule14, rule15, rule16])

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller to determine what control actions to take

        Arguments:
            ship_state (dict): contains state information for your own ship
            game_state (dict): contains state information for all objects in the game

        Returns:
            float: thrust control value
            float: turn-rate control value
            bool: fire control value. Shoots if true
            bool: mine deployment control value. Lays mine if true
        """
        # Find intercepts
        ship_pos_x = ship_state['position'][0]
        ship_pos_y = ship_state['position'][1]

        closest_asteroid = None

        # Find closest asteroid
        for a in game_state["asteroids"]:
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None or curr_dist < closest_asteroid["dist"]:
                closest_asteroid = dict(asteroid=a, dist=curr_dist)

        # If no asteroids, defaults
        if closest_asteroid is None:
             return 0, 0, False, False
        
        # Intercept calculations
        asteroid_ship_x = ship_pos_x - closest_asteroid["asteroid"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["asteroid"]["position"][1]
        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)

        asteroid_velx = closest_asteroid["asteroid"]["velocity"][0]
        asteroid_vely = closest_asteroid["asteroid"]["velocity"][1]
        asteroid_total_vel = math.sqrt(asteroid_velx**2 + asteroid_vely**2)
        asteroid_dir = math.atan2(asteroid_vely, asteroid_velx)

        theta = asteroid_ship_theta - asteroid_dir
        cos_theta = math.cos(theta)
        bullet_speed = 800

        # Time to intercept
        a = asteroid_total_vel**2 - bullet_speed**2
        b = -2 * closest_asteroid["dist"] * asteroid_total_vel * cos_theta
        c = closest_asteroid["dist"]**2
        discriminant = b**2 - 4*a*c
        bullet_t = 0
        if a != 0 and discriminant >= 0:
            t1 = (-b + math.sqrt(discriminant)) / (2*a)
            t2 = (-b - math.sqrt(discriminant)) / (2*a)
            if t1 > 0 and t2 > 0:
                bullet_t = min(t1, t2)
            elif t1 > 0:
                bullet_t = t1
            elif t2 > 0:
                bullet_t = t2

        # Find intercept point
        intercept_x = closest_asteroid["asteroid"]["position"][0] + asteroid_total_vel * bullet_t * math.cos(asteroid_dir)
        intercept_y = closest_asteroid["asteroid"]["position"][1] + asteroid_total_vel * bullet_t * math.sin(asteroid_dir)

        # Find angle to intercept point
        intercept_ship_x = intercept_x - ship_pos_x
        intercept_ship_y = intercept_y - ship_pos_y
        intercept_ship_theta = math.atan2(intercept_ship_y, intercept_ship_x)

        # Find heading theta
        shooting_theta = intercept_ship_theta - ((math.pi/180)*ship_state["heading"])

        # Normalize angle
        while shooting_theta > math.pi:
            shooting_theta -= 2*math.pi
        while shooting_theta < -1*math.pi:
            shooting_theta += 2*math.pi

        # Fuzzy inputs
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        self.simulation.input['bullet_time'] = float(min(bullet_t / 1.5, 1.0))
        self.simulation.input['theta_delta'] = float(shooting_theta)
        self.simulation.input['asteroid_dist'] = float(closest_asteroid["dist"])

        # Compute fuzzy
        self.simulation.compute()
        turn_rate = self.simulation.output['ship_turn']
        fire = True if self.simulation.output['ship_fire'] > 0 else False
        thrust = self.simulation.output['ship_thrust']
        drop_mine = True if self.simulation.output['ship_mine'] > 0 else False

        self.eval_frames += 1


        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "Test Controller"

    # @property
    # def custom_sprite_path(self) -> str:
    #     return "Neo.png"
