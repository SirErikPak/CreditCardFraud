# Introduction

Credit card fraud has shown a significant increase in attempted fraud, with e-commerce credit card fraud in the U.S. rising by 140%. Globally, 46% of credit card fraud occurs in the U.S., and global losses are projected to reach $43 billion by 2026, surpassing the GDP of many countries. Within ten years, these losses are expected to reach $397.4 billion, equivalent to the annual revenue of some Fortune 500 companies.

The statistics paint a grim picture of the widespread nature of credit card fraud. A staggering 80% of all credit cards in circulation have been compromised, and 65% of all credit and debit cardholders have been victims of fraud. The leading type of identity theft, card-not-present fraud, accounts for 65% of losses. The average credit card fraud case reported to police is $400, with a median charge of $79. Shockingly, 20% of cardholders have experienced fraud two or more times, and 150 million Americans were victims last year. For every $1 lost to online fraud, merchants incur an [actual cost](https://merchantcostconsulting.com/lower-credit-card-processing-fees/credit-card-fraud-statistics/) of $3.75. This is not just a statistic but a call to action for all of us to be vigilant. A significant portion of the population (20% globally, 10% in the U.S. annually) has been affected by compromised records or identity theft.

<!-- ## About the Dataset

This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

## Source of Simulation

This was generated using Sparkov Data Generation | Github tool created by Brandon Harris. This simulation was run for the duration - 1 Jan 2019 to 31 Dec 2020. The files were combined and converted into a standard format.

The simulator has certain pre-defined list of merchants, customers and transaction categories. And then using a python library called "faker", and with the number of customers, merchants that you mention during simulation, an intermediate list is created.

After this, depending on the profile you choose for e.g. "adults 2550 female rural.json" (which means simulation properties of adult females in the age range of 25-50 who are from rural areas), the transactions are created. Say, for this profile, you could check "Sparkov | Github | adults_2550_female_rural.json", there are parameter value ranges defined in terms of min, max transactions per day, distribution of transactions across days of the week and normal distribution properties (mean, standard deviation) for amounts in various categories. Using these measures of distributions, the transactions are generated using faker.

What I did was generate transactions across all profiles and then merged them together to create a more realistic representation of simulated transactions.

## Acknowledgements

Brandon Harris for his amazing work in creating this easy-to-use simulation tool for creating fraud transaction datasets. -->

## Feature Engineering

**Extracting Time-Based Features:**

- Quarter of the Year (trans_qtr): The numerical quarter (1-4) in which the transaction occurred.
- Month of the Year (trans_month): The numerical month (1-12) of the transaction.
- Day of the Month (trans_day): The numerical day (1-31) of the transaction.
- Day of the Week (trans_day_of_week): The numerical day of the week (Monday=0, Sunday=6) of the transaction.
- Hour of the Day (trans_hour): The numerical hour (0-23) of the transaction.
- Week of the Year (trans_week_of_year): The ISO calendar week number (1-52 or 53) of the transaction.
- Is Weekend (is_weekend): A boolean flag indicating whether the transaction occurred on a Saturday or Sunday.

**Cyclical Feature Transformation:**

For trans_hour, trans_month, and trans_day_of_week, sine and cosine transformations are applied (_sin and _cos suffixes). This is a common technique to represent cyclical features in a way that preserves their continuous and cyclical nature (e.g., 11 PM is closer to 1 AM than to 1 PM), which can be beneficial for machine learning models that don't inherently understand cyclical relationships (like linear models or tree-based models that might treat these as discrete categories).
Industry Identifier Mapping:

**Credit Card Industry Identifier:**

Map the first digit of a credit card number (Major Industry Identifier - MII) to a specific industry type (e.g., '1' for 'Airlines', '4' for 'Banking & Financial'). A new column 'industry' is created using the first digit of the cc_num (credit card number) to its corresponding industry based on the mii_to_industry mapping.

**Credit Card Network Determination:**

A new column 'cc_network' is created. This column identifies the credit card network (e.g., Visa, Mastercard).
Store Distance Calculation:

**Usage Distance:**

A new column 'store_distance' is calculated. This column represents the Haversine distance between the customer's latitude/longitude (lat, long) and the merchant's latitude/longitude (merch_lat, merch_long).

**Amount Transformation (Log Scale):**

A new column 'amt_log' is created by applying a logarithmic transformation to the amt (amount) column. This transformation is often used to reduce the skewness of heavily skewed numerical features, making them more suitable for certain machine learning models.

**Calculating Age (age):**

A new feature age is calculated based on the dob (date of birth) column and the current date (today). This correctly accounts for whether the person's birthday has passed in the current year.

## Key Findings on Fraudulent Transaction Patterns (via Chi-Square Tests)

1. **Time of Day**

- A significant association exists between transaction hours and fraud.
- Late-night hours (22:00–03:00) show disproportionately high fraud rates.
- E.g., 22:00 and 23:00 account for over 50% of fraud despite being only ~5% of non-fraud transactions.
- Daytime (04:00–21:00) sees relatively less fraud.

2. **Day of Week**

- Thursday and Friday have higher fraud rates than expected.
- Sunday and Monday show lower fraud rates relative to their total transaction volumes.
- Saturday’s fraud rate is roughly proportional to its transaction volume.

3. **Month**

- Fraud rates vary significantly across months.
- Higher fraud: January, February, March, May.
- Lower fraud: December, July, April, November.
- Suggests seasonal patterns or event-driven fraud trends.

4. **Transaction Category**

- Certain categories show elevated fraud risk:
- High-risk: grocery_pos, shopping_net, misc_net.
- Low-risk: entertainment, food_dining, health_fitness, home, travel.
- These patterns support targeted fraud prevention by category.

5. **Industry**

- Non-uniform fraud distribution across industries.
- Slightly elevated fraud in: Airlines, Banking/Financial sectors (e.g., Mastercard).
- Proportional or lower fraud in: Visa, Amex, Diners Club.

6. **Credit Card Network**

- JCB and Maestro show slightly higher fraud representation.
- Diners Club shows lower fraud involvement.
- These differences, while small, are statistically significant due to the large dataset.

7. **Gender**

- Females: More overall transactions, slightly underrepresented in fraud.
- Males: Fewer transactions, slightly overrepresented in fraud.
- Gender is a statistically relevant variable, though differences are subtle.

Chi-Square tests reveal strong associations between fraud and several variables—hour, day, month, category, industry, network, and gender—providing valuable insights for fraud detection models and risk-based strategies.

#### Discretization Age groups (20-36, 36-44, 44-53, 53-66, 66-100)

Research shows that both younger and older adults are at higher risk for fraud, but for different reasons. Younger adults tend to fall for scams more frequently, often due to less experience and a greater willingness to take risks. In comparison, older adults typically lose more money per scam, likely because of reduced vigilance and lower digital literacy. Studies highlight that both financial and digital literacy are critical protective factors against fraud. Therefore, fraud prevention efforts should be tailored: younger adults benefit from education about common scam tactics, while older adults need support systems, trusted contacts, and easy ways to report suspicious activity.

**Why Age Matters in Fraud Risk:**
The pattern aligns with broader research on fraud vulnerability. Younger adults (20-36) are often more susceptible due to less experience in recognizing scams, higher exposure to digital platforms, and a greater willingness to take risks with new financial services. They might not have developed robust habits for verifying information or protecting personal data. On the other hand, older adults (53-100)face elevated risks stemming from potential cognitive decline, increased trust in official-looking communications, and social isolation that scammers can exploit. Additionally, a lower familiarity with digital technology makes them more vulnerable to online fraud and phishing attempts.

**Tailored Fraud Prevention is Key:**
Given these distinct vulnerabilities, age-specific fraud prevention strategies are essential. For younger demographics, the focus should be on education about common scam tactics and promoting strong digital and financial literacy. For older adults, prevention efforts should include establishing support systems with trusted contacts, making reporting suspicious activity easy, and providing resources to enhance digital literacy. By understanding and addressing the unique risks associated with different age groups, fraud prevention efforts can be significantly more effective.
