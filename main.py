from models.model import Predictor

p = Predictor(db='./input/soccer.sqlite')

p.load_match_data(match_api_id=1989903)
prediction = p.predict()
p.plot()

