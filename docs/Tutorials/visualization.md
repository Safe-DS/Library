# Visualization

The following code will use a Jupyter Notebook environment.

## Table & Statistics
First, we need some data to visualize. For this, we use the common example of the titanic disaster.

!!! note
    You can download that dataset on [kaggle](https://www.kaggle.com/c/titanic).

```python
from safeds.data.tabular import Table
data = Table.from_csv("path/to/your/data.csv")
```

Now we want to have a look at what our dataset looks like. For this, we use Jupyter Notebooks native display function.

```python
data    # calls display(data)
```

![Table](./Resources/Table.png)

Next some statistics.

```python
data.summary()  # returns a table with various statistics for each column
```

![Summary](./Resources/Summary.png)

As you can see here, the **idness** of the column _PassengerId_ is 1. This means, that every row has a unique value for
this column. Since this isn't helpful for our usecase we can drop it.

```python
data_cleaned = data.drop_columns(["PassengerId"])
```
## Heatmap
Now we have a rough idea of what we are looking at. But we still don't really know a lot about our dataset.
So next we can start to plot our columns against each other in a so called Heatmap, to understand which values relate to each other.

But since this type of diagram only works for numerical values, we are going to use only those.

```python
from safeds.plotting import correlation_heatmap

data_only_numerics = Table.from_columns(data_cleaned.list_columns_with_numerical_values())
correlation_heatmap(data_only_numerics)
```

![Heatmap](./Resources/Heatmap.png)

As you can see, the columns _Fare_ and _Pclass_ (Passenger Class) seem to heavily correlate. Let's have another look at that.
## Lineplot
We'll use a lineplot to better understand their relationship.

```python
from safeds.plotting import lineplot
lineplot(data_cleaned, "Pclass", "Fare")
```

![Lineplot](./Resources/Lineplot.png)

The line itself represents the central tendency and the hued area around it a confidence interval for that estimate.

We can conclude that tickets for first classrooms are much more expensive compared to second and third class.
Also, the difference between second and third is less pronounced.

## Other plots
Some other plots that might be useful are boxplots, histograms and scatterplots.

```python
from safeds.plotting import boxplot

boxplot(data_cleaned.get_column("Age"))
```

![Boxplot](./Resources/Boxplot.png)

```python
from safeds.plotting import histogram

histogram(data_cleaned.get_column("Fare"))
```
![Histogram](./Resources/Histogram.png)

```python
from safeds.plotting import scatterplot

scatterplot(data_cleaned, "Age", "Fare")
```

![Scatterplot](./Resources/Scatterplot.png)
