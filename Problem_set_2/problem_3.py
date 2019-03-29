import numpy as np


def p_Ai_D(produced, defective):
    p_D = np.sum(produced * defective)
    p_Ai_D = (defective * produced / p_D) * 100
    return p_Ai_D


# Total produced in percentage
produced = np.array([35, 15, 5, 20, 25]) / 100
# Defective units in percentage
defective = np.array([2, 4, 10, 3.5, 3.1]) / 100
print(produced * defective * 100)

# Bayes: Probablity that defective unit came from A_i facility
print(p_Ai_D(produced, defective))


# print('Probablity device comes from A_2 give its defective %.2f' % (p_A2_D))

temp = defective * produced
temp_max = max(temp)
temp_index = np.argmax(temp)

new_defective = temp_max / produced
print(new_defective * 100)


########

produced = np.array([0.27, 0.1, 0.05, 0.08, 0.25, 0.033, 0.019, 0.085,
                     0.033, 0.02, 0.015, 0.022, 0.015, 0.008])

defective = np.array([0.02, 0.04, 0.1, 0.035, 0.022, 0.092, 0.12, 0.07,
                      0.11, 0.02, 0.07, 0.06, 0.099, 0.082])

temp = defective * produced
temp_max = max(temp)
temp_index = np.argmax(temp)

new_defective = temp_max / produced
print(np.round(new_defective, 3) )
