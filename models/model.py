import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import collections
from itertools import permutations, repeat


class Predictor(object):
	def __init__(self, db, base_area=1, leeway=None):
		self.base_area = base_area
		self.leeway = leeway
		self.predictions = []  # dataframe?
		self.accuracy = None
		self.processed = None
		self.conn = sqlite3.connect(db)

		self.match_df = None
		self.home_team_df = None
		self.away_team_df = None
		self.home_players_df = None
		self.away_players_df = None
		self.intersections = []
		self.intersection_areas = []

		self.home_squares_df = None
		self.away_squares_df = None

		self.home_team_area = None
		self.away_team_area = None
		self.intersection_area = None

		self.home_formation = None
		self.away_formation = None

	def load_match_data(self, match_api_id):
		# Load match and player data into a dataframe
		self.match_df = pd.read_sql_query(
			"SELECT * FROM match WHERE match_api_id =?", self.conn,
			params=(match_api_id,))

		# Get Team Names
		self.home_team_df = pd.read_sql_query(
			"SELECT * FROM team WHERE team_api_id = ?",
			self.conn, params=(str(self.match_df['home_team_api_id'][0]),))

		self.away_team_df = pd.read_sql_query(
			"SELECT * FROM team WHERE team_api_id = ?",
			self.conn, params=(str(self.match_df['away_team_api_id'][0]),))

		self.home_players_df = self._get_player_data()
		self.away_players_df = self._get_player_data(away=True)

		# clamp attackers' max position to the position of the home defenders
		self.away_players_df['y'] = self.away_players_df['y'].where(
			self.away_players_df['y'] != self.away_players_df['y'].min(),
			other=self.home_players_df['y'].min())

		# translate down for display purposes
		self.home_players_df['y'] = self.home_players_df['y'] - 1
		self.away_players_df['y'] = self.away_players_df['y'] - 1

	def load_match_data_many(self):
		pass

	def predict(self):
		self.home_squares_df = self._get_square_coords(self.home_players_df)
		self.away_squares_df = self._get_square_coords(self.away_players_df)
		self._calc_interesections()
		self._calc_areas()

		final_home_area = self.home_team_area - self.intersection_area
		final_away_area = self.away_team_area - self.intersection_area

		# Check area vs score
		if final_home_area > final_away_area:
			predicted_winner = "home"
		elif final_away_area > final_home_area:
			predicted_winner = "away"
		else:
			predicted_winner = "draw"

		home_goals = self.match_df['home_team_goal'][0]
		away_goals = self.match_df['away_team_goal'][0]

		if home_goals > away_goals:
			actual_winner = "home"
		elif away_goals > home_goals:
			actual_winner = "away"
		else:
			actual_winner = "draw"

		return 1 if predicted_winner == actual_winner else 0

	def predict_many(self, data):
		pass

	def _get_player_data(self, away=False):
		idx0 = 66 if away else 55
		idx1 = 77 if away else 66

		pids = self.match_df.iloc[:, idx0:idx1].values.tolist()[0]
		player_data = pd.read_sql_query(f"SELECT player_attributes.player_api_id AS pid, player.player_name AS name, (player_attributes.overall_rating / 100.0) as rating, MAX(player_attributes.date) as date \
									  FROM player_attributes \
									  LEFT JOIN player ON player.player_api_id = player_attributes.player_api_id \
									  WHERE player_attributes.player_api_id IN ({','.join(['?'] * len(pids))}) GROUP BY player_attributes.player_api_id",
										self.conn, params=(*pids,))

		player_data.drop('date', axis=1, inplace=True)

		x_idx0 = 22 if away else 11
		x_idx1 = 33 if away else 22
		y_idx0 = 44 if away else 33
		y_idx1 = 55 if away else 44

		positions = pd.DataFrame({'pid': pids,
								  'x'  : self.match_df.iloc[:,
										 x_idx0:x_idx1].values.tolist()[0],
								  'y'  : self.match_df.iloc[:,
										 y_idx0:y_idx1].values.tolist()[0]})

		# flip y coords and translate up. Note: this is a bit dodgy because the home_players_df must exist first...
		if away:
			positions['y'] = self.home_players_df['y'].max() + self.home_players_df[
				'y'].min() - positions[
								 'y']  # 14 = 11 + 3 where 11 is max_y and 3 is the min_y_defender

		player_data = player_data.merge(positions, on=('pid'))

		# drop the keeper
		player_data = player_data[
			player_data.x != 1]  # This messes with the index...

		return player_data

	def _get_formation(self, away=False):
		c = collections.Counter()
		d = self.away_players_df['y'] if away else self.home_players_df['y']
		y_coords = d.sort_values(ascending=not away)

		for item in y_coords:
			c.update(str(item))
		formation = ''
		for k, v in c.items():
			formation += '-' + str(v)
		return formation[1:]

	@staticmethod
	def _get_square_coords(players_df):
		schema = {'x0': int, 'y0': int, 'x1': int, 'y1': int, 'x2': int,
				  'y2': int, 'x3': int, 'y3': int}
		df = pd.DataFrame(columns=schema.keys()).astype(schema)

		for _, data in players_df[['x', 'y', 'rating']].iterrows():
			x = data['x']
			y = data['y']
			r = data['rating']
			df = df.append({'x0': x - r, 'y0': y - r, 'x1': x - r, 'y1': y + r,
							'x2': x + r, 'y2': y - r, 'x3': x + r, 'y3': y + r},
						   ignore_index=True)
		return df

	def _calc_interesections(self):
		# Check if each square overlaps another by comparing their max and min points in each dimension
		for i, hsq in self.home_squares_df.iterrows():
			hmaxx = hsq[['x0', 'x1', 'x2', 'x3']].max()
			hmaxy = hsq[['y0', 'y1', 'y2', 'y3']].max()
			hminx = hsq[['x0', 'x1', 'x2', 'x3']].min()
			hminy = hsq[['y0', 'y1', 'y2', 'y3']].min()

			for j, asq in self.away_squares_df.iterrows():
				# store the coordinates of the intersection in each dimension
				pair_x = []
				pair_y = []

				amaxx = asq[['x0', 'x1', 'x2', 'x3']].max()
				amaxy = asq[['y0', 'y1', 'y2', 'y3']].max()
				aminx = asq[['x0', 'x1', 'x2', 'x3']].min()
				aminy = asq[['y0', 'y1', 'y2', 'y3']].min()

				if amaxx >= hmaxx:  # if away is bigger than home
					pair_x.append(max(aminx, hminx))
					pair_x.append(hmaxx)
					l_x = hmaxx - aminx
				else:  # if home is bigger than away
					pair_x.append(max(hminx, aminx))
					pair_x.append(amaxx)
					l_x = amaxx - hminx

				if amaxy >= hmaxy:  # if away is bigger than home
					pair_y.append(max(aminy, hminy))
					pair_y.append(hmaxy)
					l_y = hmaxy - aminy
				else:  # if home is bigger than away
					pair_y.append(max(hminy, aminy))
					pair_y.append(amaxy)
					l_y = amaxy - hminy

				if l_y >= 0 and l_x >= 0:  # or equal to?
					self.intersections.append([(x, y) for x in pair_x for y in
										  pair_y])  # get the permutations of the two sets of coord pairs to get all intersection points
					self.intersection_areas.append((l_y * l_x))

	def _calc_areas(self):
		home_areas = (self.home_squares_df[['x0', 'x1', 'x2', 'x3']].max(axis=1) -
					  self.home_squares_df[['x0', 'x1', 'x2', 'x3']].min(axis=1)) * \
					 (self.home_squares_df[['y0', 'y1', 'y2', 'y3']].max(axis=1) -
					  self.home_squares_df[['y0', 'y1', 'y2', 'y3']].min(axis=1))

		away_areas = (self.away_squares_df[['x0', 'x1', 'x2', 'x3']].max(axis=1) -
					  self.away_squares_df[['x0', 'x1', 'x2', 'x3']].min(axis=1)) * \
					 (self.away_squares_df[['y0', 'y1', 'y2', 'y3']].max(axis=1) -
					  self.away_squares_df[['y0', 'y1', 'y2', 'y3']].min(axis=1))

		self.home_team_area = home_areas.sum()
		self.away_team_area = away_areas.sum()
		self.intersection_area = sum(self.intersection_areas)

	def plot(self):
		plt.rc('figure', dpi=100)
		plt.rc('grid', linestyle="-", color='black')
		plt.rc('figure', figsize=(10, 10))
		plt.xlim(0, 11)
		plt.ylim(0, 11)
		plt.grid(True)
		plt.xticks(range(0, 11))
		plt.yticks(range(0, 11))

		# TODO: fix get_formation
		self.home_formation = self._get_formation()
		self.away_formation = self._get_formation(away=True)

		# Plot positions on grid
		plt.scatter(self.home_players_df['x'], self.home_players_df['y'], s=50, c='blue',
					label=f'[H] {self.home_team_df["team_short_name"][0]} {self.home_formation}')
		plt.scatter(self.away_players_df['x'], self.away_players_df['y'], s=50, c='red',
					label=f'[A] {self.away_team_df["team_short_name"][0]} {self.away_formation}')

		self._draw_area_of_influence(self.home_players_df, 'blue')
		self._draw_area_of_influence(self.away_players_df, 'red')

		# plt.axis('scaled')

		# Legend
		plt.legend()
		plt.title(
			f"{self.home_team_df['team_long_name'][0]} {self.match_df['home_team_goal'][0]} - {self.match_df['away_team_goal'][0]} {self.away_team_df['team_long_name'][0]}")

		# Intersections
		for i in self.intersections:
			x, y = zip(*i)
			plt.scatter(x, y, s=20, c='purple', marker='x')

		plt.show()

	@staticmethod
	def _draw_area_of_influence(df, color):
		for _, data in df[['x', 'y', 'rating']].iterrows():
			influence = data['rating'] * 2
			x_rect = data['x'] - influence / 2
			y_rect = data['y'] - influence / 2
			square = plt.Rectangle((x_rect, y_rect), influence, influence,
								   fc=color, alpha=0.2)
			plt.gca().add_patch(square)

