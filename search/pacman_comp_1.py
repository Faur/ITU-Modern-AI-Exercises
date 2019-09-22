## run with
# >> python pacman_comp_1.py -n 10 -g DirectionalGhost

import csv
import importlib
import numpy as np

from pacman import *
import textDisplay

students = {
    'tokf': np.nan,
    'your_student_name': np.nan,
    'another_student': np.nan,
    'more_students': np.nan,
}

if __name__ == '__main__':
    args = readCommand( sys.argv[1:] ) # Get game components based on input
    args['display'] = textDisplay.NullGraphics()

    for key in students:
        try:
            module = importlib.import_module(key+'.CompetitionPacman')
            args['pacman'] = module.CompAgent()
            # exec('from key import CompetitionPacman')
            out = runGames( **args)
            scores = [o.state.getScore() for o in out]

            students[key] = np.mean(scores)
        except ImportError as e:
            print('Error with', key)
            print(e)


print('')
print('#'*50)
print('#'*50)
print('#'*50)
print('')

for key in students:
    print(key, students[key])

# Save results to a csv file
w = csv.writer(open("student_scores.csv", "w"))
for key, val in students.items():
    w.writerow([key, val])

print('!!')