# Targeted Strategies in Reducing the Spread of COVID-19 and Future Pandemics

This web application examines the differences in the effectiveness of state-mandated COVID-19 interventions among U.S. counties, and how they relate to certain county characteristics. The project goal is to find more targeted approaches to handling future respiratory-based pandemics that would address specific needs of each community. 

### Identify county subgroups via k-means clustering

The first step is to divide the U.S. counties into subgroups based on the following community-level characteristics: 
-	Population density (2018)
-	Percent in poverty
-	Percentage of seniors in population

With these characteristics, preliminary analysis with k-means clustering will create 3 different subgroups of counties.  


## Evaluate effectiveness of COVID-19 policies among each subgroup.

The second step is to evaluate the effectiveness of COVID-19 policies among each subgroup. We will use a 7-day moving average of positive case percentage as a measure of COVID-19 spread. For this application, positive case percentage refers to (# daily new positive test cases) / (1000 county residents). We will calculate this measure at two time points: the date of intervention, and  3 weeks after implementation. Our outcome variable will be the difference in these two measurements. The policy interventions we will consider at this stage are the following: 
-	Mask Mandates
-	Business Closure Orders
-	Stay at Home Orders


## Data Sources

The data used for this application come from the following sources: 
•	The HPC Data Hub at Hopkins Population Center:
	https://popcenter.jhu.edu/data-hub/
•	U.S. Department of Commerce, Bureau of the Census, Small Area Income and Poverty Estimates (SAIPE) Program:
https://www.census.gov/programs-surveys/saipe.html
•	COVID-19 US State Policy Database:
https://www.openicpsr.org/openicpsr/project/119446/version/V44/view

