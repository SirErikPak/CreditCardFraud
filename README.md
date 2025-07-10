## Introduction

Credit card fraud has shown a significant increase in attempted fraud, with e-commerce credit card fraud in the U.S. rising by 140%. Globally, 46% of credit card fraud occurs in the U.S., and global losses are projected to reach $43 billion by 2026, surpassing the GDP of many countries. Within ten years, these losses are expected to reach $397.4 billion, equivalent to the annual revenue of some Fortune 500 companies.

The statistics paint a grim picture of the widespread nature of credit card fraud. A staggering 80% of all credit cards in circulation have been compromised, and 65% of all credit and debit cardholders have been victims of fraud. The leading type of identity theft, card-not-present fraud, accounts for 65% of losses. The average credit card fraud case reported to police is $400, with a median charge of $79. Shockingly, 20% of cardholders have experienced fraud two or more times, and 150 million Americans were victims last year. For every $1 lost to online fraud, merchants incur an [actual cost](https://merchantcostconsulting.com/lower-credit-card-processing-fees/credit-card-fraud-statistics/) of $3.75. This is not just a statistic but a call to action for all of us to be vigilant. A significant portion of the population (20% globally, 10% in the U.S. annually) has been affected by compromised records or identity theft.

<!-- ## About the Dataset

This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

## Source of Simulation

This was generated using Sparkov Data Generation | Github tool created by Brandon Harris. This simulation was run for the duration - 1 Jan 2019 to 31 Dec 2020. The files were combined and converted into a standard format.

The simulator has certain pre-defined list of merchants, customers and transaction categories. And then using a python library called "faker", and with the number of customers, merchants that you mention during simulation, an intermediate list is created.

After this, depending on the profile you choose for e.g. "adults 2550 female rural.json" (which means simulation properties of adult females in the age range of 25-50 who are from rural areas), the transactions are created. Say, for this profile, you could check "Sparkov | Github | adults_2550_female_rural.json", there are parameter value ranges defined in terms of min, max transactions per day, distribution of transactions across days of the week and normal distribution properties (mean, standard deviation) for amounts in various categories. Using these measures of distributions, the transactions are generated using faker.

What I did was generate transactions across all profiles and then merged them together to create a more realistic representation of simulated transactions.

## Acknowledgements -->

<!-- Brandon Harris for his amazing work in creating this easy-to-use simulation tool for creating fraud transaction datasets. -->

## Feature Engineering

Extracting Time-Based Features:

- Quarter of the Year (trans_qtr): The numerical quarter (1-4) in which the transaction occurred.
- Month of the Year (trans_month): The numerical month (1-12) of the transaction.
- Day of the Month (trans_day): The numerical day (1-31) of the transaction.
- Day of the Week (trans_day_of_week): The numerical day of the week (Monday=0, Sunday=6) of the transaction.
- Hour of the Day (trans_hour): The numerical hour (0-23) of the transaction.
- Week of the Year (trans_week_of_year): The ISO calendar week number (1-52 or 53) of the transaction.
- Is Weekend (is_weekend): A boolean flag indicating whether the transaction occurred on a Saturday or Sunday.

Cyclical Feature Transformation:

For trans_hour, trans_month, and trans_day_of_week, sine and cosine transformations are applied (_sin and _cos suffixes). This is a common technique to represent cyclical features in a way that preserves their continuous and cyclical nature (e.g., 11 PM is closer to 1 AM than to 1 PM), which can be beneficial for machine learning models that don't inherently understand cyclical relationships (like linear models or tree-based models that might treat these as discrete categories).

Calculating Age (age):

A new feature age is calculated based on the dob (date of birth) column and the current date (today). This correctly accounts for whether the person's birthday has passed in the current year.
