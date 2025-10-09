python -u asgn3/main.py | python asgn3/ai_console_filter.py -i "WARNING: Nan, Inf or huge value in QACC at DOF \d+. The simulation is unstable. Time = 0.\d+." | python asgn3/ai_console_filter.py "^$"
