# macroeconomic-random-forest
playground to play with merf, llms and streamlit

Looks like with python 3.13 and the required libraries that some work was needed to correct MRF.py. This is now kept in fixed_code and needs to be analysed and, perhaps, submitted in a PR.

# todo
1. download simple data - done
2. need to clean the data - ensure that it's all on a consistent monthly basis - done need to think about seasonality and normalising the data.
3. need to model just three variables using the MRF framework and see if it works with the changes actioned - employment, inflation, interest rates vs gdp and include government spending as the exogenous variable done
3a. the bands don't make sense - need to analyse that code and see what it's doing
3b. plot the actual data versus the mrf modelling
3c. yep the variables aren't enough think the docs oversell what this can do?
3c.a. leaving modelling to one side as llms are going to help with this
4. get simple streamlit ui to select and preprocess the variables of interest done
4a. hwo does it use pyarrow?
4b. run as 
```streamlit run simple_streamlit.py```
4c. now need help preprocessing hte data
4c.a. ask gpt directly about the data and how to preprocess it
4c.a.1 convert absolute values to relative and add as columns
4c.a.2 use interpolation
4c.a.3 lagged variables
4c.a.4 model up until pandemic as it throws all the figures off
5. plot graph of actual versus mrf *done
5a. get bands.
6. One we have the bands we need to ask llm to comment on them
