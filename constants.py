SCHEMA = ['epochseconds',
          'player_a_id', 'player_a_rank', 'player_a_mmr', 'player_a_race',
          'player_b_id', 'player_b_rank', 'player_b_mmr', 'player_b_race',
          'map', 'turnrate', 'latency', 'uptime', 'outcome']
SCHEMA_LEN = 14
assert SCHEMA_LEN == len(SCHEMA)
OPPONENT_RACE_IDX = 8
RESULT_IDX = SCHEMA_LEN - 1
ARTOSIS_NAME_IDX = 1
OPPONENT_NAME_IDX = 5
ARTOSIS_MMR_IDX = 3
OPPONENT_MMR_IDX = 7
TURNRATE_IDX = 10

NAME_PATTERNS = {'artosis': 'artosis',
                 'arto': 'artosis',
                 'valks': 'artosis',
                 'canadadry': 'artosis',
                 'artasis': 'artosis',
                 'newgear': 'artosis',
                 'didntmake': 'artosis'}

MAP_PATTERNS = {'polypoid': 'polypoid',
                'potypoid': 'polypoid',
                'poly': 'polypoid',
                'eclipse': 'eclipse',
                'clipse': 'eclipse',
                'good night': 'goodnight',
                'good': 'goodnight',
                'largo': 'largo',
                'larg': 'largo',
                'ood night': 'goodnight'}
