import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def main():
    st.title("House Price Estimation ")
    st.sidebar.title("House Price Estimation Web App")
    st.markdown("What is the price of your favourite house ? 🏠")
    st.sidebar.markdown("What is the price of your favourite house ? 🏠")

    # @st.cache(persist=True)
    # def load_data():
    house = pd.read_csv("house_usa.csv")
    house_final = house[
        ['MSZoning', 'LotArea', 'Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual', 'YearBuilt', 'ExterQual',
         'TotalBsmtSF',
         'HeatingQC', 'FullBath', 'BedroomAbvGr', 'KitchenQual', 'GarageType', 'TotRmsAbvGrd', 'GarageArea',
         'PoolArea', 'SalePrice']]
    le = LabelEncoder()
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()
    le5 = LabelEncoder()
    le6 = LabelEncoder()
    dit = {}
    # for i in house_final.columns:
    #     if i in ['MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle', 'ExterQual', 'HeatingQC', 'KitchenQual',
    #              'GarageType']:
    gd = {'2Types': 0, 'Attchd': 1, 'Basment': 2, 'BuiltIn': 3, 'CarPort': 4, 'Detchd': 5}
    house_final['MSZoning'] = le.fit_transform(house_final['MSZoning'])
    house_final['Neighborhood'] = le1.fit_transform(house_final['Neighborhood'])
    house_final['BldgType'] = le2.fit_transform(house_final['BldgType'])
    house_final['HouseStyle'] = le3.fit_transform(house_final['HouseStyle'])
    house_final['ExterQual'] = le4.fit_transform(house_final['ExterQual'])
    house_final['HeatingQC'] = le5.fit_transform(house_final['HeatingQC'])
    house_final['KitchenQual'] = le6.fit_transform(house_final['KitchenQual'])
    house_final['GarageType'] = house_final['GarageType'].replace({'2Types':0, 'Attchd':1, 'Basment':2, 'BuiltIn':3, 'CarPort':4, 'Detchd':5})
    dit = {'BldgType': ['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE'],
     'ExterQual': ['Ex', 'Fa', 'Gd', 'TA'],
     'GarageType': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd'],
     'HeatingQC': ['Ex', 'Fa', 'Gd', 'Po', 'TA'],
     'HouseStyle': ['1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story',
            'SFoyer', 'SLvl'],
     'KitchenQual': ['Ex', 'Fa', 'Gd', 'TA'],
     'MSZoning': ['C (all)', 'FV', 'RH', 'RL', 'RM'],
     'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
    'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
    'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown',
    'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber',
    'Veenker']}



    #@st.cache(persist=True)
    # def split(df):
    #     y = df.type
    #     x = df.drop(columns=['type'])
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    #     return x_train, x_test, y_train, y_test

    # def plot_metrics(metrics_list):
    #     if 'Confusion Matrix' in metrics_list:
    #         st.subheader("Confusion Matrix")
    #         plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
    #         st.pyplot()
    #
    #     if 'ROC Curve' in metrics_list:
    #         st.subheader("ROC Curve")
    #         plot_roc_curve(model, x_test, y_test)
    #         st.pyplot()
    #
    #     if 'Precision-Recall Curve' in metrics_list:
    #         st.subheader('Precision-Recall Curve')
    #         plot_precision_recall_curve(model, x_test, y_test)
    #         st.pyplot()

    classes_dict, df = dit,house_final
    class_names = ['edible', 'poisonous']

    #x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Select your features")
    mszoning = st.sidebar.selectbox("Type of land",
                                      tuple(classes_dict['MSZoning']))
    l = []
    l.append(mszoning)

    print(np.array(mszoning).shape)
    mszoning = le.transform(np.array(l))
    print(mszoning[0])

    lot = st.sidebar.number_input("Lot Area",1300,215245, step=100, key='C_SVM')

    neighbor = st.sidebar.selectbox("Neighborhood",
                                    tuple(classes_dict['Neighborhood']))
    m = []
    m.append(neighbor)
    neighbor = le1.transform(np.array(m))

    bldgtype = st.sidebar.selectbox("Building Type",
                                    tuple(classes_dict['BldgType']))
    n = []
    n.append(bldgtype)
    bldgtype = le2.transform(np.array(n))

    house_style = st.sidebar.selectbox("House Style",
                                    tuple(classes_dict['HouseStyle']))
    o = []
    o.append(house_style)
    house_style = le3.transform(np.array(o))

    quality = st.sidebar.selectbox("Overall Quality",
                                      tuple(list(range(0,11))))

    year_built = st.sidebar.selectbox("Year Built",
                                      tuple(list(range(1872,2011))))

    exter_qual = st.sidebar.selectbox("Exterior Quality",
                                       tuple(classes_dict['ExterQual']))
    p = []
    p.append(exter_qual)
    exter_qual = le4.transform(np.array(p))

    bsmt_area = st.sidebar.number_input("Basement Area", 0.0, 6110.0, step= 50.0, key='C')

    heat = st.sidebar.selectbox("Heating ",
                                      tuple(classes_dict['HeatingQC']))
    q = []
    q.append(heat)
    heat = le5.transform(np.array(q))

    total_rooms = st.sidebar.selectbox("Total rooms",
                                       tuple(list(range(2, 15))))

    num_baths = st.sidebar.selectbox("Number of bathrooms",
                                      tuple(list(range(0, 4))))

    num_bedrooms = st.sidebar.selectbox("Number of bedrooms",
                                     tuple(list(range(0, 9))))

    kitchen = st.sidebar.selectbox("Quality of Kitchen",
                                       tuple(classes_dict['KitchenQual']))
    r = []
    r.append(kitchen)
    kitchen = le6.transform(np.array(r))

    garage_type = st.sidebar.selectbox("Garage Type",
                                       tuple(classes_dict['GarageType']))
    garage_type = gd[garage_type]

    garage_area = st.sidebar.number_input("Garage Area", 0.0, 1418.0, step=50.0, key='C_S')

    pool_area = st.sidebar.selectbox("Pool Area",
                                     tuple([0, 512, 648, 576, 555, 480, 519, 738]))

    total_area = house_final.LotArea + house.GarageArea + house_final.PoolArea + house_final.TotalBsmtSF
    total_rooms_adv = house_final.FullBath + house_final.BedroomAbvGr + house_final.TotRmsAbvGrd
    quality_adv = house_final.OverallQual + house_final.ExterQual + house_final.KitchenQual
    dic = house_final.groupby("YearBuilt").mean().LotArea.to_dict()
    Year_Area_mean = house_final.YearBuilt.map(dic)
    Year_Over_median = house_final.YearBuilt.map(
        house_final.groupby("YearBuilt").median().OverallQual.to_dict())
    d = house_final.groupby(["Neighborhood", "MSZoning", "BldgType", "HouseStyle"]).mean().LotArea.to_dict()
    #four_lot =d.get((neighbor[0],mszoning[0],bldgtype[0],house_style[0]))
    e = house_final.groupby(["Neighborhood", "MSZoning", "BldgType", "HouseStyle"]).median().OverallQual.to_dict()
    #four_qual = e.get((neighbor[0],mszoning[0],bldgtype[0],house_style[0]))
    lot = np.power(lot,0.2)
    bsmt_area = np.power(bsmt_area,0.2)
    garage_area = np.power(garage_area,0.2)
    #saleprice
    year_built = np.power(year_built,3.5)
    Year_Area_mean = np.power(Year_Area_mean,0.2)
    #four_lot = np.power(four_lot,0.2)


    featues = np.array([mszoning[0],lot,neighbor[0],bldgtype[0],house_style[0],quality,year_built,exter_qual[0],bsmt_area,heat[0],
                        num_baths,num_bedrooms,kitchen[0],garage_type,total_rooms,garage_area,pool_area])#,total_area,
                       # total_rooms_adv,quality_adv,Year_Area_mean,Year_Over_median,four_lot,four_qual])
    #st.write(featues[6])
    print(featues.reshape(1,17).shape)
    #model = joblib.load("FinalPipeline.joblib")
    #st.write(featues)
    pk = open("classifier.pkl", 'rb')
    m = joblib.load("classifier.pkl")
    price = np.power(m.predict(featues.reshape(1,17)),5)
    st.write("Predicted price of the house is : $",price[0])
    st.line_chart(m.feature_importances_)
    st.line_chart(df.corr())



    #st.write(model.predict(featues))

    # if classifier == 'Support Vector Machine (SVM)':
    #     st.sidebar.subheader("Model Hyperparameters")
    #     # choose parameters
    #     C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
    #     kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
    #     gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
    #
    #     metrics = st.sidebar.multiselect("What metrics to plot?",
    #                                      ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    #
    #     if st.sidebar.button("Classify", key='classify'):
    #         st.subheader("Support Vector Machine (SVM) Results")
    #         model = SVC(C=C, kernel=kernel, gamma=gamma)
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy: ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics)
    #
    # if classifier == 'Logistic Regression':
    #     st.sidebar.subheader("Model Hyperparameters")
    #     C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
    #     max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
    #
    #     metrics = st.sidebar.multiselect("What metrics to plot?",
    #                                      ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    #
    #     if st.sidebar.button("Classify", key='classify'):
    #         st.subheader("Logistic Regression Results")
    #         model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy: ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics)
    #
    # if classifier == 'Random Forest':
    #     st.sidebar.subheader("Model Hyperparameters")
    #     n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10,
    #                                            key='n_estimators')
    #     max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
    #     bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
    #     metrics = st.sidebar.multiselect("What metrics to plot?",
    #                                      ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    #
    #     if st.sidebar.button("Classify", key='classify'):
    #         st.subheader("Random Forest Results")
    #         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
    #                                        n_jobs=-1)
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy: ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("USA House Prices")
        st.write(df)
    if st.sidebar.checkbox("Data descriptions", False):
        st.subheader("USA House Prices")
        st.markdown(
"MSZoning: Identifies the general zoning classification of the sale.\n"
		
       "A	Agriculture\n"
       "C	Commercial\n"
       "FV	Floating Village Residential\n"
       "I	Industrial\n"
       "RH	Residential High Density"
       "RL	Residential Low Density"
       "RP	Residential Low Density Park "
       "RM	Residential Medium Density"
                    
                    
"Neighborhood: Physical locations within Ames city limits"

       "Blmngtn	Bloomington Heights"
       "Blueste	Bluestem"
       "BrDale	Briardale"
       "BrkSide	Brookside"
       "ClearCr	Clear Creek"
       "CollgCr	College Creek"
       "Crawfor	Crawford"
       "Edwards	Edwards"
       "Gilbert	Gilbert"
       "IDOTRR	Iowa DOT and Rail Road"
       "MeadowV	Meadow Village"
       "Mitchel	Mitchell"
       "Names	North Ames"
       "NoRidge	Northridge"
       "NPkVill	Northpark Villa"
       "NridgHt	Northridge Heights"
       "NWAmes	Northwest Ames"
       "OldTown	Old Town"
       "SWISU	South & West of Iowa State University"
       "Sawyer	Sawyer"
       "SawyerW	Sawyer West"
       "Somerst	Somerset"
       "StoneBr	Stone Brook"
       "Timber	Timberland"
       "Veenker	Veenker"
                    
                    
"BldgType: Type of dwelling"
		
       "1Fam	Single-family Detached	"
       "2FmCon	Two-family Conversion; originally built as one-family dwelling"
       "Duplx	Duplex"
       "TwnhsE	Townhouse End Unit"
       "TwnhsI	Townhouse Inside Unit"
	
                    
"HouseStyle: Style of dwelling"
	
       "1Story	One story"
       "1.5Fin	One and one-half story: 2nd level finished"
       "1.5Unf	One and one-half story: 2nd level unfinished"
       "2Story	Two story"
       "2.5Fin	Two and one-half story: 2nd level finished"
       "2.5Unf	Two and one-half story: 2nd level unfinished"
       "SFoyer	Split Foyer"
       "SLvl	Split Level"
	
                    
"OverallQual: Rates the overall material and finish of the house"

       "10	Very Excellent"
       "9	Excellent"
       "8	Very Good"
       "7	Good"
       "6	Above Average"
       "5	Average"
       "4	Below Average"
       "3	Fair"
       "2	Poor"
       "1	Very Poor"
                   
                    
"YearBuilt: Original construction date"
                    
                    
"ExterQual: Evaluates the quality of the material on the exterior "
		
       "Ex	Excellent"
       "Gd	Good"
       "TA	Average/Typical"
       "Fa	Fair"
       "Po	Poor"
                    
"TotalBsmtSF: Total square feet of basement area"

                    
"HeatingQC: Heating quality and condition"

       "Ex	Excellent"
       "Gd	Good"
       "TA	Average/Typical"
       "Fa	Fair"
       "Po	Poor"
                    
"FullBath: Full bathrooms above grade"
                    
                    
 "KitchenQual: Kitchen quality"

       "Ex	Excellent"
       "Gd	Good"
       "TA	Typical/Average"
       "Fa	Fair"
       "Po	Poor"
       	
"TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)"
                    
"FireplaceQu: Fireplace quality"

       "Ex	Excellent - Exceptional Masonry Fireplace"
       "Gd	Good - Masonry Fireplace in main level"
       "TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement"
       "Fa	Fair - Prefabricated Fireplace in basement"
       "Po	Poor - Ben Franklin Stove"
       "NA	No Fireplace"
		
"GarageType: Garage location"
		
       "2Types	More than one type of garage"
       "Attchd	Attached to home"
       "Basment	Basement Garage"
       "BuiltIn	Built-In (Garage part of house - typically has room above garage)"
       "CarPort	Car Port"
       "Detchd	Detached from home"
       "NA	No Garage"
                    
                    
"GarageQual: Garage quality"

       "Ex	Excellent"
       "Gd	Good"
       "TA	Typical/Average"
       "Fa	Fair"
       "Po	Poor"
       "NA	No Garage"
		
"GarageCond: Garage condition"

       "Ex	Excellent"
       "Gd	Good"
       "TA	Typical/Average"
       "Fa	Fair"
       "Po	Poor"
       "NA	No Garage"
                    
                    
"PoolArea: Pool area in square feet")
#         st.markdown(
#             "This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms "
#             "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, "
#             "or of unknown edibility and not recommended. This latter class was combined with the poisonous one.")


if __name__ == '__main__':
    main()


