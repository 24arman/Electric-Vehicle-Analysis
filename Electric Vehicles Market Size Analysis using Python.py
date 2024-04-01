#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\hplap\Downloads\EV-data\Electric_Vehicle_Population_Data.csv")


# In[2]:


print(df.head(10))


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df = df.dropna()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")



plt.figure(figsize =(12,10))

ev_drop_by_year =  df['Model Year'].value_counts().sort_index()
sns.barplot (x = ev_drop_by_year.index , y = ev_drop_by_year.values , palette = 'viridis')
plt.title = ("EV Adoption Over Time")
plt.xlabel = ("Model Year")

plt.ylabel = ("Number of Vehicles Registered")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()


# In[17]:


ev = df['County'].value_counts()
top_countries = ev.head(3).index

top_countries_data = df[df['County'].isin(top_countries)]
ev1 = top_countries_data.groupby(['County', 'City']).size().sort_values(ascending = False).reset_index(name = 'Number of Vehicles')

top_cities = ev1.head(10)

figure = plt.figure(figsize =(12,10))
ax = figure.add_subplot(111)
sns.barplot(x = "Number of Vehicles", y = 'City' , hue = 'County' , data = top_cities , palette = 'magma', ax=ax)
ax.set_title("Top Cities in Top Counties by EV Registration")
ax.set_xlabel('Number of Vehicles Registered')
ax.set_ylabel('City')
ax.legend(title = 'County')
plt.tight_layout()
plt.show()


# In[19]:


ev_type_distribution = df['Electric Vehicle Type'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=ev_type_distribution.values, y=ev_type_distribution.index, palette="rocket")
ax.set_title('Distribution of Electric Vehicle Types')
ax.set_xlabel('Number of Vehicles Registered')
ax.set_ylabel('Electric Vehicle Type')
plt.tight_layout()
plt.show()


# In[21]:


ev_make_distribution = df['Make'].value_counts().head(10)  # Limiting to top 10 for clarity

plt.figure(figsize=(12, 6))
sns.barplot(x=ev_make_distribution.values, y=ev_make_distribution.index, palette="cubehelix")
ax.set_title('Top 10 Popular EV Makes')
ax.set_xlabel('Number of Vehicles Registered')
ax.set_ylabel('Make')
plt.tight_layout()
plt.show()


# In[26]:


top_3_makes = ev_make_distribution.head(3).index

top_makes_data = df[df['Make'].isin(top_3_makes)]

ev_model_distribution_top_makes = top_makes_data.groupby(['Make', 'Model']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')


top_models = ev_model_distribution_top_makes.head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='Number of Vehicles', y='Model', hue='Make', data=top_models, palette="viridis")
ax.set_title('Top Models in Top 3 Makes by EV Registrations')
ax.set_xlabel('Number of Vehicles Registered')
ax.set_ylabel('Model')
plt.legend(title='Make', loc='center right')
plt.tight_layout()
plt.show()


# In[34]:


ev = df['Electric Range'].mean()

plt.figure(figsize=(12, 6))
sns.histplot(df['Electric Range'], bins=30, kde=True, color='royalblue')
ax.set_title('Distribution of Electric Vehicle Ranges')
ax.set_xlabel('Electric Range (miles)')
ax.set_ylabel('Number of Vehicles')
plt.axvline(ev, color='red', linestyle='--', label=f'Mean Range: {ev:.2f} miles')
plt.legend()
plt.show()


# In[37]:


average_range_by_year = df.groupby('Model Year')['Electric Range'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Model Year', y='Electric Range', data=average_range_by_year, marker='o', color='green')
ax.set_title('Average Electric Range by Model Year')
ax.set_xlabel('Model Year')
ax.set_ylabel('Average Electric Range (miles)')
plt.grid(True)
plt.show()


# In[39]:


average_range_by_model = top_makes_data.groupby(['Make', 'Model'])['Electric Range'].mean().sort_values(ascending=False).reset_index()


top_range_models = average_range_by_model.head(10)

plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='Electric Range', y='Model', hue='Make', data=top_range_models, palette="cool")
ax.set_title('Top 10 Models by Average Electric Range in Top Makes')
ax.set_xlabel('Average Electric Range (miles)')
ax.set_ylabel('Model')
plt.legend(title='Make', loc='center right')
plt.show()


# In[41]:


ev_registration_model = df['Model Year'].value_counts().sort_index()
ev_registration_model


# In[43]:


from scipy.optimize import curve_fit
import numpy as np

filtered_years = ev_registration_model[ev_registration_model.index <= 2023]


def exp_growth(x, a, b):
    return a * np.exp(b * x)

x_data = filtered_years.index - filtered_years.index.min()
y_data = filtered_years.values

params, covariance = curve_fit(exp_growth, x_data, y_data)


forecast_years = np.arange(2024, 2024 + 6) - filtered_years.index.min()
forecasted_values = exp_growth(forecast_years, *params)


forecasted_evs = dict(zip(forecast_years + filtered_years.index.min(), forecasted_values))

print(forecasted_evs)


# In[45]:


years = np.arange(filtered_years.index.min(), 2029 + 1)
actual_years = filtered_years.index
forecast_years_full = np.arange(2024, 2029 + 1)

actual_values = filtered_years.values
forecasted_values_full = [forecasted_evs[year] for year in forecast_years_full]

plt.figure(figsize=(12, 8))
plt.plot(actual_years, actual_values, 'bo-', label='Actual Registrations')
plt.plot(forecast_years_full, forecasted_values_full, 'ro--', label='Forecasted Registrations')

ax.set_title('Current & Estimated EV Market')
ax.set_xlabel('Year')
ax.set_ylabel('Number of EV Registrations')
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:


# market size analysis is a crucial aspect of market research that determines the potential sales volume within a given market. It helps businesses understand the magnitude of demand, assess market saturation levels, and identify growth opportunities. From our market size analysis of electric vehicles, we found a promising future for the EV industry, indicating a significant shift in consumer preferences and a potential increase in related investment and business opportunities.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




