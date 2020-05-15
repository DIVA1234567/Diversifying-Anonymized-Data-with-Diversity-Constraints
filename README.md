# Datasets

## Pantheon Dataset
This dataset is from [source](https://pantheon.world/explore/rankings?show=people), which contains 11,341 individuals' information, based
on the popularity of their biographical page in Wikipedia . 

| **Attribute Details**                                                                                                                                                                                                                                              |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| article_id                                                                                                                                                                                                                                                    |
| full_name                                                                                                                                                                                                                                                                                  |
| sex                                                                                                                                                                                                                                                                                                      |
| birth_year                                                                                                                                                                                                                                                                                 |
| city: The city in which the figure was primarily active.                                                                                                                                              |
| state: The state (if applicable) in which the figure was primarily active.                                                                                          |
| country: The country in which the figure was primarily active.                                                                                                                    |
| continent: The continent in which the figure was primarily active.                                                                                                        |
| latitude: The central latitude of the city, state, etc. that the figure was active in.                                                             |
| longitude: The central longitude of the city, state, etc. that the figure was active in.                                                 |
| occupation: The historical figure's specific occupation. E.g. physicist.                                                                                              |
| industry: The industry (or topic area) the figure concentrated their work in. E.g. philosophy.                          |
| domain: The general area of contribution the figure is known for. E.g. humanities.                                                     |
| article_languages: How many of the different language Wikipedias have an article on the figure.     |
| page_views: The estimated total page views for the figure across all Wikipedias.                                                            |
| average_views: The estimated average page views for the figure per each Wikipedia edition article. |
| historical_popularity_index: An index value measuring approximately the popularity                                                    |

## Census Dataset 
 The dataset contains 300K records with 40 attributes from the U.S. Census Bureau. This dataset can be found here [source](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.html), containing population data.

| **Attribute Details** |
| :------------------------------------------------ |
|age: continuous.|
|workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.|
|education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, |
|Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.|
|education-num: continuous.|
|marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.|
|occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, |
|Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.|
|relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.|
|race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.|
|sex: Female, Male.|
|capital-gain: continuous.|
|capital-loss: continuous.|
|hours-per-week: continuous.|
|native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.|

## German Credit Dataset
This dataset can be found here [source](https://archive.ics.uci.edu/ml/datasets/), which contains individuals' credit report information.  

| **Attribute Details** |
| :------------------------------------------------ |
|Age|
|Sex|
|Job: unskilled and non-resident, unskilled and resident, skilled, high|
|Housing: own, rent, free|
|Saving accounts: little, moderate, quite rich, rich|
|Checking account|
|Credit amount|
|Duration|
|Purpose: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others|

## Diversity Constraints

Constraints
Given a diversity constraint, it has four components: attribute name, attribute value, the frequency of the attribute value (lower bound) and the frequency of the attribute value (lower bound). We denote in this format `attrName(attrValue)[lower bound, upper bound]`. For example, `city(New York) [100, 400]` means the attribute name is city, the attribute value New York. the lower bound 100 and upper bound 400.

Diversity constraints examples:

```
city(New York) [100, 400]
city(Paris) [100, 300]
city(Los Angeles) [100, 200]


country(United States) [100, 150]
country(United Kingdom) [100, 150]
country(France) [100, 150]


continent(Europe) [100, 150]
continent(North America) [100, 150]


occupation(Politician) [1000, 2500]
occupation(Actor) [600, 1500]


industry (Government) [1000, 2500]
industry (Film And Theatre) [600, 1500]
```

Diversity Constraints

### Diversity Constraints Defined on Pantheon

Diversity constraints on attributes Sex and Continent

```
Minimum:
Sex(Male) [1, 5636]
Sex(Female) [1, 1]

continent(Europe) [1, 5631]
continent(North America) [1, 1]
continent(Asia) [1, 1]
continent(Africa) [1, 1]
continent(South America) [1, 1]
continent(Oceania) [1, 1]

Average:
Sex(Male) [2818, 2818]
Sex(Female) [2818, 2818]

continent(Europe) [939, 2850]
continent(North America) [939, 939]
continent(Asia) [939, 939]
continent(Africa) [419, 419]
continent(South America) [366, 366]
continent(Oceania) [123, 123]


Proportion
Sex(Male) [4893, 4893]
Sex(Female) [742, 742]

continent(Europe) [3164, 3164]
continent(North America) [1212, 1212]
continent(Asia) [590, 590]
continent(Africa) [208, 208]
continent(South America) [181, 181]
continent(Oceania) [61, 61]

domain(Institutions) [1728, 1728]
domain(Arts) [1433, 1433]
domain(Sports) [887, 887]
domain(Science & Technology) [688, 688]
domain(Humanities) [664, 664]
domain(Public Figure) [179, 179]
domain(Business & Law) [54, 54]
domain(Exploration) [51, 51]
```



### Diversity Constraints Defined on Census

Diversity constraints on attributes Sex and Race

```
Minimum:

Sex(Male) [1, 286882]
Sex(Female) [1, 1]

race(White)[1, 286882]
race(Black)[1, 1]
race(Asian-Pac-Islander)[1, 1]
race(Amer-Indian-Eskimo)[1, 1]
race(Other)[1, 1]

Average:

Sex(Male) [217900, 286882]
Sex(Female) [107710, 143441]

race(White)[57376, 286882]
race(Black)[31240, 31240]
race(Asian-Pac-Islander)[10390, 10390]
race(Amer-Indian-Eskimo)[3110, 3110]
race(Other)[2710, 2710]

Proportion

sex(Male) [208868, 286882]
sex(Female) [103246, 143441]

race(White)[266631, 286882]
race(Black)[29945, 29945]
race(Asian-Pac-Islander)[9959, 9959]
race(Amer-Indian-Eskimo)[2981, 2981]
race(Other)[2598, 2598]

sex(Male)race(White)[186331, 286882]
sex(Male)race(Black)[19700, 19700]
sex(Male)race(Asian-Pac-Islander)[5411, 5411]
sex(Male)race(Amer-Indian-Eskimo)[1994, 1994]
sex(Male)race(Other)[1844, 1844]

sex(Female)race(White)[80300, 80300]
sex(Female)race(Black)[10245, 10245]
sex(Female)race(Asian-Pac-Islander)[4548, 4548]
sex(Female)race(Amer-Indian-Eskimo)[987, 987]
sex(Female)race(Other)[754, 754]

workclass(private)[184521, 234511]
workclass(other)[64151, 64151]
workclass(gov)[87451, 87451]
```

### Diversity Constraints Defined on Credit

Diversity constraints on attributes Sex and Job

```
Minimum:

Sex(Male) [1, 690]
Sex(Female) [1, 310]

Job(2)[1, 630]
Job(1)[1, 200]
Job(3)[1, 148]
Job(0)[1, 22]


Average:

Sex(Male) [500, 690]
Sex(Female) [310, 500]

Job(2)[250, 630]
Job(1)[200, 200]
Job(3)[148, 148]
Job(0)[22, 22]


Proportion

Sex(Male) [662, 690]
Sex(Female) [298, 298]

Job(2)[605, 605]
Job(1)[192, 192]
Job(3)[142, 142]
Job(0)[21, 21]
```


## Source Code
You can find Source Code of this paper [here](https://bitbucket.org/YuHuangMac/diva/src/master/)
