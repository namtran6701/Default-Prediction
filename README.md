if missing vals is less than 1 percent, drop 
5->10% impute with median, mode
more than that, consider dropping the cols

Categorical notes: 
term: two terms only - 36 and 60 months -> Convert to 36:0, 60:1

grade: 7 grades - A to G -> Convert to A:0, B:1, C:2, D:3, E:4, F:5, G:6

sub_grade: 35 subgrades - A1 to G5 -> Convert to A1:0, A2:1, A3:2, A4:3, A5:4, B1:5, B2:6, B3:7, B4:8, B5:9, C1:10, C2:11, C3:12, C4:13, C5:14, D1:15, D2:16, D3:17, D4:18, D5:19, E1:20, E2:21, E3:22, E4:23, E5:24, F1:25, F2:26, F3:27, F4:28, F5:29, G1:30, G2:31, G3:32, G4:33, G5:34

home_ownership: 6 types - MORTGAGE, RENT, OWN, OTHER, NONE, ANY -> Convert to MORTGAGE:0, RENT:1, OWN:2, OTHER:3, NONE:4, ANY:5

verification_status: 3 types - Not Verified, Source Verified, Verified -> Convert to Not Verified:0, Source Verified:1, Verified:2

purpose: 14 types - credit_card, debt_consolidation, car, home_improvement, major_purchase, small_business, other, medical, moving, vacation, house, renewable_energy, wedding, educational -> Convert to credit_card:0, debt_consolidation:1, car:2, home_improvement:3, major_purchase:4, small_business:5, other:6, medical:7, moving:8, vacation:9, house:10, renewable_energy:11, wedding:12, educational:13

addr_state: 51 states -> Convert to 51 columns with 1s and 0s

application_type: 2 types - INDIVIDUAL, JOINT -> Convert to INDIVIDUAL:0, JOINT:1

pymnt_plan: 2 types - n, y -> Convert to n:0, y:1

initial_list_status: 2 types - f, w -> Convert to f:0, w:1


Numerical notes: 

emp_length: 11 types - less than 1 year, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10+ -> Convert to 0:less than 1 year, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10+:10

int_rate - convert to float

revol_util - convert to float

earliest_cr_line - convert to number of months from earliest credit line to the date of loan application

last_pymnt_d - convert to number of months from last payment date to the date of loan application

last_credit_pull_d - convert to number of months from last credit pull date to the date of loan application

issue_d - convert to number of months from issue date to the date of loan application




