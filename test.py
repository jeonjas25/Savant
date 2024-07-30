from pybaseball import  statcast_pitcher
kershaw_stats = statcast_pitcher('2017-06-01', '2017-07-01', 477132)
kershaw_stats.groupby("pitch_type").release_speed.agg("mean")