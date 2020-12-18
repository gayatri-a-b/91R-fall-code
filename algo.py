import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import wasserstein_distance
from matplotlib.animation import FuncAnimation


# Threshold Classifier Agent
class Threshold_Classifier:

    # constructor
    def __init__( self, 
                  pi_0 = [10, 10, 20, 30, 30, 0, 0],
                  pi_1 = [0, 10, 10, 20, 30, 30, 0],
                  certainty_0 = [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7],
                  certainty_1 = [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7],
                  loan_chance = 0.1,
                  group_chance = 0.5,
                  loan_amount = 1.0,
                  interest_rate = 0.3,
                  bank_cash = 1000 ):
    
        self.pi_0 = pi_0 # disadvantaged group
        self.certainty_0 = certainty_0
        self.pi_1 = pi_1 # advantaged group
        self.certainty_1 = certainty_1

        # chance you loan to someone you are uncertain whether they will repay
        self.loan_chance = loan_chance

        # chance they come group 1
        self.group_chance = group_chance

        # loan amount
        self.loan_amount = loan_amount

        # interest rate
        self.interest_rate = interest_rate

        # bank cash
        self.bank_cash = bank_cash


    # return an individual and their outcome
    def get_person(self):
        # what group they are in
        group = numpy.random.choice(2, 1, p=[1 - self.group_chance, self.group_chance])[0]

        # what their credit score bin they walk in from
        decile = numpy.random.choice(7, 1, p=[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])[0]
        
        # whether they will repay or not
        if group == 0:
            loan = numpy.random.choice(2, 1, p=[1 - self.certainty_0[decile], self.certainty_0[decile]])[0]
        else:
            loan = numpy.random.choice(2, 1, p=[1 - self.certainty_1[decile], self.certainty_1[decile]])[0]

        # determine whether to loan to uncertain person
        if loan == 0:
            loan = numpy.random.choice(2, 1, p=[1 - self.loan_chance, self.loan_chance])[0]

        # determine whether they repay or not
        if group == 0:
            repay = numpy.random.choice(2, 1, p=[1 - self.certainty_0[decile], self.certainty_0[decile]])[0]
        else:
            repay = numpy.random.choice(2, 1, p=[1 - self.certainty_1[decile], self.certainty_1[decile]])[0]

        # if nobody in that bin
        if group == 0 and self.pi_0[decile] == 0 :
            loan = 0 
        if group and self.pi_1[decile] == 0:
            loan = 0

        return ((group, decile, loan, repay))


    # get the average score of a given distribution
    def average_score(self, pi_):
        average = ( 1*pi_[0] + 2*pi_[1] + 3*pi_[2] + 4*pi_[3] + 5*pi_[4] + 6*pi_[5] + 7*pi_[6] ) / 100
        return average


    # return what an update of the environment would yield
    def one_step_update(self):
        # make copies of then current variables
        pi_0 = self.pi_0.copy()
        certainty_0 = self.certainty_0.copy()
        pi_1 = self.pi_1.copy()
        certainty_1 = self.certainty_1.copy()
        bank_cash = self.bank_cash

        # get the person and their outcome
        (group, decile, loan, repay) = self.get_person()

        # if group 0
        if group == 0:
            # if we loaned
            if loan:
                current_bin = pi_0[decile] # current bin count
                current_repayment = certainty_0[decile] # current bin repayment certainty

                # if loan was not repaid
                if repay == 0:

                    # if they can be moved down
                    if decile != 0:
                        bin_under = pi_0[decile - 1] # bin under count
                        repayment_under = certainty_0[decile - 1] # bin under repayment certainty

                        # update count of current bin
                        pi_0[decile] = pi_0[decile] - 1
                        # update count of bin under
                        pi_0[decile - 1] = pi_0[decile - 1] + 1

                    # bank loses money
                    bank_cash -= self.loan_amount
                # if loan was repaid
                else:

                    # if they can be moved down
                    if decile != 6:
                        bin_over = pi_0[decile + 1] # bin under count
                        repayment_over = certainty_0[decile + 1] # bin under repayment certainty

                        # update count of current bin
                        pi_0[decile] = pi_0[decile] - 1
                        # update count of bin over
                        pi_0[decile + 1] = pi_0[decile + 1] + 1

                    # bank gains money
                    bank_cash += self.loan_amount * (1 + self.interest_rate)

            return (group, pi_0, certainty_0, bank_cash, loan, repay)              
        
        # if group 1
        else:
            # if we loaned
            if loan:
                current_bin = pi_1[decile] # current bin count
                current_repayment = certainty_1[decile] # current bin repayment certainty

                # if loan was not repaid
                if repay == 0:

                    # if they can be moved down
                    if decile != 0:
                        bin_under = pi_1[decile - 1] # bin under count
                        repayment_under = certainty_1[decile - 1] # bin under repayment certainty

                        # update count of current bin
                        pi_1[decile] = pi_1[decile] - 1
                        # update count of bin under
                        pi_1[decile - 1] = pi_1[decile - 1] + 1

                    # bank loses money
                    bank_cash -= self.loan_amount
                # if loan was repaid
                else:

                    # if they can be moved down
                    if decile != 6:
                        bin_over = pi_1[decile + 1] # bin under count
                        repayment_over = certainty_1[decile + 1] # bin under repayment certainty

                        # update count of current bin
                        pi_1[decile] = pi_1[decile] - 1
                        # update count of bin over
                        pi_1[decile + 1] = pi_1[decile + 1] + 1

                    # bank gains money
                    bank_cash += self.loan_amount * (1 + self.interest_rate)
            
            return (group, pi_1, certainty_1, bank_cash, loan, repay)


    # take one step of the environment if successful iteration
    # return whether successful update took place
    def one_step(self):
        # get current distribution averages
        current_average_0 = self.average_score(self.pi_0)
        current_average_1 = self.average_score(self.pi_1)

        # check out what one step would be
        (group, pi_, certainty_, bank_cash_, loan_, repay_) = self.one_step_update()

        # get the proposed step distribution
        potential_average_ = self.average_score(pi_)

        # if group 0
        if group == 0:
            # get the wasserstein distance
            earth_mover_distance_ = wasserstein_distance(self.pi_0, pi_)

            # successful step means average increased and bank at least breaks even
            if bank_cash_ >= 1000 and potential_average_ >= current_average_0:
                # take the step
                self.pi_0 = pi_
                self.certainty_0 = certainty_
                self.bank_cash = bank_cash_

                # successful update
                return True

        # if group 1
        else:
            # get the wasserstein distance
            earth_mover_distance_ = wasserstein_distance(self.pi_1, pi_)

            # successful step means average increased and bank at least breaks even
            if bank_cash_ >= 1000 and potential_average_ >= current_average_1:                
                # take the step
                self.pi_1 = pi_
                self.certainty_1 = certainty_
                self.bank_cash = bank_cash_

                # successful update
                return True

        # not successful so no update took place
        return False


    # update specific number of times
    def iterate(self, iterations):
        # index
        i = 0

        # only count successful updates
        while (i < iterations):
            self.one_step()
            i += 1

        # return distributions after given number of successful updates
        return (self.pi_0, self.pi_1)



## SIMULATION
# parameter values from: https://github.com/google/ml-fairness-gym/blob/master/environments/lending_params.py
pi_0 = [10, 10, 20, 30, 30, 0, 0] # Disadvantaged group distribution
pi_1 = [0, 10, 10, 20, 30, 30, 0] # Advantaged group distribution
certainty_0 = [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7] # Likelihood of repayment by credit score (fixed during simulation)
certainty_1 = [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7] # Likelihood of repayment by credit score (fixed during simulation)
group_chance = 0.5 # chance of selecting a group (fixed during simulation)
loan_amount = 1.0 # amount of each loan (fixed during simulation)
interest_rate = 0.3 # interest rate of loans (fixed during simulation)
bank_cash = 1000 # starting amount in bank -- altruistic bank so seeks to a least break even

# tunable hyper parameters
loan_chance = 0.02 # chance of loaning to someone who should not receive the loan
iterations = 300 # number of time steps to simulate


"""
## TO RUN SIMULATION
# RUN
th = Threshold_Classifier(
        pi_0 = pi_0,
        pi_1 = pi_1,
        certainty_0 = certainty_0,
        certainty_1 = certainty_1,
        loan_chance = loan_chance,
        group_chance = group_chance,
        loan_amount = loan_amount,
        interest_rate = interest_rate,
        bank_cash = bank_cash )
(updated_pi_0, updated_pi_1) = th.iterate(iterations)


# PRINT
# print distribution before and after
print("Time steps: ", iterations)
print("Inital Group A (Disadvantaged): ", pi_0)
print("Updated Group A (Disadvantaged): ", updated_pi_0)
print("Inital Group B (Advantaged): ", pi_1)
print("Updated Group B (Advantaged): ", updated_pi_1)
"""

"""
## TEST ITERATIONS
# test iterations
iterations_array = [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 10000]

ones_0 = []
twos_0 = []
threes_0 = []
fours_0 = []
fives_0 = []
sixes_0 = []
sevens_0 = []

ones_1 = []
twos_1 = []
threes_1 = []
fours_1 = []
fives_1 = []
sixes_1 = []
sevens_1 = []

updated_pi_0s = []
updated_pi_1s = []

for iteration_i in iterations_array:
    iterations = iteration_i

    th = Threshold_Classifier(
        pi_0 = pi_0,
        pi_1 = pi_1,
        certainty_0 = certainty_0,
        certainty_1 = certainty_1,
        loan_chance = loan_chance,
        group_chance = group_chance,
        loan_amount = loan_amount,
        interest_rate = interest_rate,
        bank_cash = bank_cash )

    (updated_pi_0, updated_pi_1) = th.iterate(iterations)

    updated_pi_0s.append(updated_pi_0)
    updated_pi_1s.append(updated_pi_1)

    ones_0.append(updated_pi_0[0])
    twos_0.append(updated_pi_0[1])
    threes_0.append(updated_pi_0[2])
    fours_0.append(updated_pi_0[3])
    fives_0.append(updated_pi_0[4])
    sixes_0.append(updated_pi_0[5])
    sevens_0.append(updated_pi_0[6])

    ones_1.append(updated_pi_1[0])
    twos_1.append(updated_pi_1[1])
    threes_1.append(updated_pi_1[2])
    fours_1.append(updated_pi_1[3])
    fives_1.append(updated_pi_1[4])
    sixes_1.append(updated_pi_1[5])
    sevens_1.append(updated_pi_1[6])



data = {'Iterations': iterations_array, 
        '100s': ones_0, '200s': twos_0, '300s': threes_0, '400s': fours_0, '500s': fives_0, '600s': sixes_0, '700s': sevens_0,
        '100s-1': ones_1, '200s-1': twos_1, '300s-1': threes_1, '400s-1': fours_1, '500s-1': fives_1, '600s-1': sixes_1, '700s-1': sevens_1}
df = pandas.DataFrame(data=data)

df.to_csv(r'data-iterations.csv', index = False)


data_0 = {'Iterations': iterations_array, 
        '100s': ones_0, '200s': twos_0, '300s': threes_0, '400s': fours_0, '500s': fives_0, '600s': sixes_0, '700s': sevens_0}
df_0 = pandas.DataFrame(data=data_0)

data_1 = {'Iterations': iterations_array,
        '100s-1': ones_1, '200s-1': twos_1, '300s-1': threes_1, '400s-1': fours_1, '500s-1': fives_1, '600s-1': sixes_1, '700s-1': sevens_1}
df_1 = pandas.DataFrame(data=data_1)


for ind, arr in enumerate(updated_pi_0s):
    iterations_ = iterations_array[ind]

    objects = ('100', '200', '300', '400', '500', '600', '700')
    y_pos = numpy.arange(len(objects))
    outcome_ = arr

    plt.bar(y_pos, outcome_, align='center', alpha=0.5)

    plt.xticks(y_pos, objects)
    plt.ylabel('Number of People')
    plt.xlabel('Credit Score')
    plt.title('Group 0 ' + str(iterations_) + " Iterations")
    plt.savefig('charts-iterations/group_0_' + str(iterations_) + '.png')
    plt.clf()

for ind, arr in enumerate(updated_pi_1s):
    iterations_ = iterations_array[ind]

    objects = ('100', '200', '300', '400', '500', '600', '700')
    y_pos = numpy.arange(len(objects))
    outcome_ = arr

    plt.bar(y_pos, outcome_, align='center', alpha=0.5)

    plt.xticks(y_pos, objects)
    plt.ylabel('Number of People')
    plt.xlabel('Credit Score')
    plt.title('Group 1 ' + str(iterations_) + " Iterations")
    plt.savefig('charts-iterations/group_1_' + str(iterations_) + '.png')
    plt.clf()
"""


"""
## TEST LOAN_CHANGE
# test loan chance
loan_chance_array = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]

ones_0 = []
twos_0 = []
threes_0 = []
fours_0 = []
fives_0 = []
sixes_0 = []
sevens_0 = []

ones_1 = []
twos_1 = []
threes_1 = []
fours_1 = []
fives_1 = []
sixes_1 = []
sevens_1 = []

updated_pi_0s = []
updated_pi_1s = []

for loan_chance_i in loan_chance_array:
    loan_chance = loan_chance_i

    th = Threshold_Classifier(
        pi_0 = pi_0,
        pi_1 = pi_1,
        certainty_0 = certainty_0,
        certainty_1 = certainty_1,
        loan_chance = loan_chance,
        group_chance = group_chance,
        loan_amount = loan_amount,
        interest_rate = interest_rate,
        bank_cash = bank_cash )

    (updated_pi_0, updated_pi_1) = th.iterate(iterations)

    updated_pi_0s.append(updated_pi_0)
    updated_pi_1s.append(updated_pi_1)

    ones_0.append(updated_pi_0[0])
    twos_0.append(updated_pi_0[1])
    threes_0.append(updated_pi_0[2])
    fours_0.append(updated_pi_0[3])
    fives_0.append(updated_pi_0[4])
    sixes_0.append(updated_pi_0[5])
    sevens_0.append(updated_pi_0[6])

    ones_1.append(updated_pi_1[0])
    twos_1.append(updated_pi_1[1])
    threes_1.append(updated_pi_1[2])
    fours_1.append(updated_pi_1[3])
    fives_1.append(updated_pi_1[4])
    sixes_1.append(updated_pi_1[5])
    sevens_1.append(updated_pi_1[6])



data = {'Loan-Chance': loan_chance_array, 
        '100s': ones_0, '200s': twos_0, '300s': threes_0, '400s': fours_0, '500s': fives_0, '600s': sixes_0, '700s': sevens_0,
        '100s-1': ones_1, '200s-1': twos_1, '300s-1': threes_1, '400s-1': fours_1, '500s-1': fives_1, '600s-1': sixes_1, '700s-1': sevens_1}
df = pandas.DataFrame(data=data)

df.to_csv(r'data-loan-chance.csv', index = False)


data_0 = {'Loan-Chance': loan_chance_array, 
        '100s': ones_0, '200s': twos_0, '300s': threes_0, '400s': fours_0, '500s': fives_0, '600s': sixes_0, '700s': sevens_0}
df_0 = pandas.DataFrame(data=data_0)

data_1 = {'Loan-Chance': loan_chance_array,
        '100s-1': ones_1, '200s-1': twos_1, '300s-1': threes_1, '400s-1': fours_1, '500s-1': fives_1, '600s-1': sixes_1, '700s-1': sevens_1}
df_1 = pandas.DataFrame(data=data_1)


for ind, arr in enumerate(updated_pi_0s):
    loan_chance_ = loan_chance_array[ind]

    objects = ('100', '200', '300', '400', '500', '600', '700')
    y_pos = numpy.arange(len(objects))
    outcome_ = arr

    plt.bar(y_pos, outcome_, align='center', alpha=0.5)

    plt.xticks(y_pos, objects)
    plt.ylabel('Number of People')
    plt.xlabel('Credit Score')
    plt.title('Group 0 ' + str(loan_chance_) + " Loan Chance")
    plt.savefig('charts-loan-chance/group_0_' + str(loan_chance_) + '.png')
    plt.clf()

for ind, arr in enumerate(updated_pi_1s):
    loan_chance_ = loan_chance_array[ind]

    objects = ('100', '200', '300', '400', '500', '600', '700')
    y_pos = numpy.arange(len(objects))
    outcome_ = arr

    plt.bar(y_pos, outcome_, align='center', alpha=0.5)

    plt.xticks(y_pos, objects)
    plt.ylabel('Number of People')
    plt.xlabel('Credit Score')
    plt.title('Group 1 ' + str(loan_chance_) + " Loan Chance")
    plt.savefig('charts-loan-chance/group_1_' + str(loan_chance_) + '.png')
    plt.clf()
"""