import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt

st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title='True Data Visualize',  # String or None. Strings get appended with "• Streamlit". 
	page_icon='./t1.png',  # String, anything supported by st.image, or None.
)

st.image('./true.png')

st.sidebar.header('User Input')
dataview1 = st.sidebar.selectbox('Data View Option', ['RR_Fault_Trendline','District_Fault_Table'])

up_file = st.file_uploader('Upload your data file', type=["txt","xlsx","csv"])
if up_file is not None:
    st.write('You selected `%s`' % up_file.name)
else:
    st.write('You selected `%s`' % up_file)

if up_file is not None:
    df = pd.read_table(up_file)

    if dataview1 == 'RR_Fault_Trendline':

        province =st.sidebar.selectbox('Province Option',df.SCAB_PROVINCE_E.unique())

        c1, c2,c3= st.beta_columns((1, 1,3))
        with c1:
            rr_select = st.selectbox('View Option', ['Month','Week'])
        with c2:
            if rr_select == 'Week':
                wk_value = st.selectbox('Choose Week',df.FAULT_COMPLETE_WEEK.unique())

        if rr_select == 'Month':
            allticket = df[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','FAULT_FAULT_TYPE_CD','FAULT_DEPARTMENT_GROUP','FAULT_DEPARTMENT','RR_TRUCK_FAULT_IS_RR30DAY','RR_ALL_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE','FAULT_COMPLETE_WEEK']]
            allticket = allticket[allticket.SCAB_PROVINCE_E == province]
        else:
            allticket = df[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','FAULT_FAULT_TYPE_CD','FAULT_DEPARTMENT_GROUP','FAULT_DEPARTMENT','RR_TRUCK_FAULT_IS_RR30DAY','RR_ALL_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE','FAULT_COMPLETE_WEEK']]
            allticket = allticket[allticket.SCAB_PROVINCE_E == province]
            allticket = allticket[allticket.FAULT_COMPLETE_WEEK == wk_value]

        ##########      RR Truck Roll      ###########
        
        rr = df[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','FAULT_DEPARTMENT_GROUP','RR_TRUCK_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE','FAULT_COMPLETE_WEEK']]
        rr['RR_TRUCK_FAULT_IS_RR30DAY'] = rr['RR_TRUCK_FAULT_IS_RR30DAY'].replace(np.nan, 0)
        r30 = pd.get_dummies(rr.RR_TRUCK_FAULT_IS_RR30DAY)
        rr['RR'] = r30.RR
        if rr_select == 'Month':
            nrt = rr[(rr.SCAB_PROVINCE_E == province) & (rr.FAULT_DEPARTMENT_GROUP == 'Truck Roll') & (rr.FAULT_TICKET_TYPE == 'Regular Fault')]
        else:
            nrt = rr[(rr.SCAB_PROVINCE_E == province) & (rr.FAULT_DEPARTMENT_GROUP == 'Truck Roll') & (rr.FAULT_TICKET_TYPE == 'Regular Fault')]
            nrt = nrt[nrt.FAULT_COMPLETE_WEEK == wk_value]

        nrt.reset_index(drop=True, inplace=True)
	st.write('#Noted : Click at the top Right Botton to view in Full Screen')
        st.write("Truck Roll Data :")
        nrt

        new_rr = allticket.groupby(['FAULT_COMPLETE_DATE'])[['FAULT_TICKET_TYPE']].count()
        new_rr.rename(columns={'FAULT_TICKET_TYPE':'All Tiket Count'},inplace = True)

        new_rr['FAULT_TICKET_NUMBER'] = nrt.groupby(['FAULT_COMPLETE_DATE'])[['FAULT_TICKET_NUMBER']].count()
        new_rr['FAULT_TICKET_NUMBER'] = new_rr['FAULT_TICKET_NUMBER'].replace(np.nan, 0)
        new_rr['RR'] = nrt.groupby(['FAULT_COMPLETE_DATE'])[['RR']].sum()
        new_rr['RR'] = new_rr['RR'].replace(np.nan, 0)
        accu_f = list(accumulate(new_rr.iloc[:,1]))
        new_rr['Accumulated Fault'] = accu_f
        accu_rr = list(accumulate(new_rr.iloc[:,2]))
        new_rr['Accumulated RR'] = accu_rr
        percent = round(new_rr.iloc[:,4] / new_rr.iloc[:,3] *100, 2)
        new_rr['RR Rate Accumulate(%)'] = percent
        st.write('RR Truck Roll Report : ')
        tran_rr = new_rr.transpose()
        tran_rr
        
        chart_new_rr = new_rr
        chart_new_rr.reset_index(drop=False, inplace=True)

        Y=chart_new_rr['RR Rate Accumulate(%)']
        X=chart_new_rr.index
        # regression
        reg = LinearRegression().fit(np.vstack(X), Y)
        chart_new_rr['bestfit'] = reg.predict(np.vstack(X))

        fig3 = px.bar(chart_new_rr, y='RR Rate Accumulate(%)', x='FAULT_COMPLETE_DATE',text ='RR Rate Accumulate(%)',title="Truck Roll RR Rate(%)" )
        fig3.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig3.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig3, use_container_width=True)


        fig2 = go.Figure(data=go.Scatter(name='RR Rate',x = X, y = Y.values))
        fig2.add_trace(go.Scatter(name='line of best fit', x=X, y=chart_new_rr['bestfit'], mode='lines'))
        for x,y in zip(X,Y):
            a = '(Day:%s,' %(x+1)
            a = a+'%s)' %y
            fig2.add_annotation(x=x, y=y,
                text=a,
                showarrow=False,
                yshift=10)
        fig2.update_layout(showlegend=False)
        fig2.update_layout(title="Truck Roll RR Rate(%)", xaxis_title = 'Day', yaxis_title = 'RR Rate(%)')
        st.plotly_chart(fig2, use_container_width=True)


        #############################################

        ##########      RR All Fault      ###########


        allrr = df[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','RR_ALL_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE','FAULT_COMPLETE_WEEK']]
        allrr['RR_ALL_FAULT_IS_RR30DAY'] = allrr['RR_ALL_FAULT_IS_RR30DAY'].replace(np.nan, 0)
        allr30 = pd.get_dummies(allrr.RR_ALL_FAULT_IS_RR30DAY)
        allrr['RR'] = allr30.RR

        if rr_select == 'Month':
            all_nrt = allrr[(allrr.SCAB_PROVINCE_E == province) & (allrr.FAULT_TICKET_TYPE == 'Regular Fault')]
        else:
            all_nrt = allrr[(allrr.SCAB_PROVINCE_E == province) & (allrr.FAULT_TICKET_TYPE == 'Regular Fault')]
            all_nrt = all_nrt[all_nrt.FAULT_COMPLETE_WEEK == wk_value]

        all_nrt.reset_index(drop=True, inplace=True)
	st.write('#Noted : Click at the top Right Botton to view in Full Screen')
        st.write("All Type Data :")
        all_nrt
        all_new_rr = allticket.groupby(['FAULT_COMPLETE_DATE'])[['FAULT_TICKET_TYPE']].count()
        all_new_rr.rename(columns={'FAULT_TICKET_TYPE':'All Tiket Count'},inplace = True)

        all_new_rr['FAULT_TICKET_NUMBER'] = all_nrt.groupby(['FAULT_COMPLETE_DATE'])[['FAULT_TICKET_NUMBER']].count()
        all_new_rr['FAULT_TICKET_NUMBER'] = all_new_rr['FAULT_TICKET_NUMBER'].replace(np.nan, 0)
        all_new_rr['RR'] = all_nrt.groupby(['FAULT_COMPLETE_DATE'])[['RR']].sum()
        all_new_rr['RR'] = all_new_rr['RR'].replace(np.nan, 0)
        all_accu_f = list(accumulate(all_new_rr.iloc[:,1]))
        all_new_rr['Accumulated Fault'] = all_accu_f
        all_accu_rr = list(accumulate(all_new_rr.iloc[:,2]))
        all_new_rr['Accumulated RR'] = all_accu_rr
        all_percent = round(all_new_rr.iloc[:,4] / all_new_rr.iloc[:,3] *100, 2)
        all_new_rr['RR Rate Accumulate(%)'] = all_percent
        st.write('RR All Fault Report : ')
        all_tran_rr = all_new_rr.transpose()
        all_tran_rr
        
            
        all_chart_new_rr = all_new_rr
        all_chart_new_rr.reset_index(drop=False, inplace=True)

        all_Y=all_chart_new_rr['RR Rate Accumulate(%)']
        all_X=all_chart_new_rr.index
        # regression
        all_reg = LinearRegression().fit(np.vstack(all_X), all_Y)
        all_chart_new_rr['bestfit'] = all_reg.predict(np.vstack(all_X))


        all_fig3 = px.bar(all_chart_new_rr, y='RR Rate Accumulate(%)', x='FAULT_COMPLETE_DATE',text ='RR Rate Accumulate(%)',title="All Type RR Rate(%)" )
        all_fig3.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        all_fig3.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(all_fig3, use_container_width=True)

        all_fig2 = go.Figure(data=go.Scatter(name='RR Rate',x = all_X, y = all_Y.values))
        all_fig2.add_trace(go.Scatter(name='line of best fit', x=all_X, y=all_chart_new_rr['bestfit'], mode='lines'))
        for x,y in zip(all_X,all_Y):
            a = '(Day:%s,' %(x+1)
            a = a+'%s)' %y
            all_fig2.add_annotation(x=x, y=y,
                text=a,
                showarrow=False,
                yshift=10)
        all_fig2.update_layout(showlegend=False)
        all_fig2.update_layout(title="All Type RR Rate(%)", xaxis_title = 'Day', yaxis_title = 'RR Rate(%)')
        st.plotly_chart(all_fig2, use_container_width=True)

        #############################################

    if dataview1 == 'District_Fault_Table':

        province =st.sidebar.selectbox('Province Option',df.SCAB_PROVINCE_E.unique())

        district_table = df[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','FAULT_FAULT_TYPE_CD','FAULT_DEPARTMENT_GROUP','FAULT_DEPARTMENT','RR_TRUCK_FAULT_IS_RR30DAY','RR_ALL_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE']]
        district_table = district_table[district_table.SCAB_PROVINCE_E == province]
        district_table['RR_TRUCK_FAULT_IS_RR30DAY'] = district_table['RR_TRUCK_FAULT_IS_RR30DAY'].replace(np.nan, 0)
        district_table['RR_ALL_FAULT_IS_RR30DAY'] = district_table['RR_ALL_FAULT_IS_RR30DAY'].replace(np.nan, 0)
        district_table.reset_index(drop=True, inplace=True)

        ############### TRUCK ROLL ##################
        d1 = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
        d1.reset_index(drop=True, inplace=True)

        district = district_table.groupby(['SCAB_DISTRICT_E'])[['FAULT_TICKET_TYPE']].count()
        district.rename(columns={'FAULT_TICKET_TYPE':'All Tiket Count'},inplace = True)
        district['Total TT Closed'] = d1.groupby(['SCAB_DISTRICT_E'])[['RR_TRUCK_FAULT_IS_RR30DAY']].count()

        d1rr = pd.get_dummies(d1.RR_TRUCK_FAULT_IS_RR30DAY)
        d1['RR_Truck'] = d1rr.RR

        district['Total TT 30 Day'] = d1.groupby(['SCAB_DISTRICT_E'])[['RR_Truck']].sum()
        district_type = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll')]
        district_type.reset_index(drop=True, inplace=True)

        d_type = pd.get_dummies(district_type.FAULT_FAULT_TYPE_CD)
        district_type['066'] = d_type['066 | Proactive']
        district_type['275'] = d_type['275 | ไฟ Los ติด เป็นสีแดง']
        district_type['307'] = d_type['307 | Proactive-Fiber broken']
        district_type['371'] = d_type['371 | Proactive - Fiber degrade']

        district['Total TT 066'] = district_type.groupby(['SCAB_DISTRICT_E'])[['066']].sum()
        district['Total TT 275'] = district_type.groupby(['SCAB_DISTRICT_E'])[['275']].sum()
        district['Total TT 307'] = district_type.groupby(['SCAB_DISTRICT_E'])[['307']].sum()
        district['Total TT 371'] = district_type.groupby(['SCAB_DISTRICT_E'])[['371']].sum()

        district['Total TT Closed'] = district['Total TT Closed'].replace(np.nan, 0)
        district['Total TT 30 Day'] = district['Total TT 30 Day'].replace(np.nan, 0)

        d_percent = round(district.iloc[:,2] / district.iloc[:,1] *100, 2)
        district['%RR30Day'] = d_percent
        district['%RR30Day'] = district['%RR30Day'].replace(np.nan, 0)
        

        ########## FILTER TO FIND RR [TRUCK ROLL]#############

        d1_rnum = district_table[(district_table.RR_TRUCK_FAULT_IS_RR30DAY == 'RR') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
        d1_rnum.reset_index(drop=True, inplace=True)

        ######################################################
        ########################### RR#1 #########################

        value_counts1 = d1_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 1]
        df_value_counts1 = pd.DataFrame(value_counts1)
        df_value_counts1.columns = ['RR#1']
        df_value_counts1.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
        district['RR#1'] = df_value_counts1.groupby('SCAB_DISTRICT_E')['RR#1'].count()

        ##########################################################

        ########################### RR#2 #########################

        value_counts2 = d1_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 2]
        df_value_counts2 = pd.DataFrame(value_counts2)
        df_value_counts2.columns = ['RR#2']
        df_value_counts2.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
        district['RR#2'] = df_value_counts2.groupby('SCAB_DISTRICT_E')['RR#2'].count()

        ##########################################################

        ########################### RR#3 #########################

        value_counts3 = d1_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 3]
        df_value_counts3 = pd.DataFrame(value_counts3)
        df_value_counts3.columns = ['RR#3']
        df_value_counts3.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
        district['RR#3'] = df_value_counts3.groupby('SCAB_DISTRICT_E')['RR#3'].count()

        ##########################################################

        ########################### RR>=4 #########################

        value_counts4 = d1_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x >= 4]
        df_value_counts4 = pd.DataFrame(value_counts4)
        df_value_counts4.columns = ['RR>=4']
        df_value_counts4.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
        district['RR>=4'] = df_value_counts4.groupby('SCAB_DISTRICT_E')['RR>=4'].count()

        ##########################################################

        district['RR#1'] = district['RR#1'].replace(np.nan, 0)
        district['RR#2'] = district['RR#2'].replace(np.nan, 0)
        district['RR#3'] = district['RR#3'].replace(np.nan, 0)
        district['RR>=4'] = district['RR>=4'].replace(np.nan, 0)


        district.loc['Grand Total']= district.sum(numeric_only=True, axis=0)
        d12_percent = round(district.iloc[-1,2] / district.iloc[-1,1] *100, 2)
        district.iloc[-1,7] = d12_percent

        st.write('Truck Roll Accumulated:')
        st.write('#Noted : Click at the top Right Botton to view in Full Screen')
        district

        # Colors
        # st.table(
        #     district.style.applymap(color_negative_red).apply(
        #         highlight_max, color="yellow", axis=0
        #     )
        # )

        ###################################################################################

        #################### ALL FAULT #########################

        d2 = district_table[(district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
        d2.reset_index(drop=True, inplace=True)

        district_all = district_table.groupby(['SCAB_DISTRICT_E'])[['FAULT_TICKET_TYPE']].count()
        district_all.rename(columns={'FAULT_TICKET_TYPE':'All Tiket Count'},inplace = True)
        district_all['Total TT Closed'] = d2.groupby(['SCAB_DISTRICT_E'])[['RR_ALL_FAULT_IS_RR30DAY']].count()

        d2rr = pd.get_dummies(d2.RR_ALL_FAULT_IS_RR30DAY)
        d2['All_Truck'] = d2rr.RR

        district_all['Total TT 30 Day'] = d2.groupby(['SCAB_DISTRICT_E'])[['All_Truck']].sum()
        district_type = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll')]
        district_type.reset_index(drop=True, inplace=True)

        d_type = pd.get_dummies(district_type.FAULT_FAULT_TYPE_CD)
        district_type['066'] = d_type['066 | Proactive']
        district_type['275'] = d_type['275 | ไฟ Los ติด เป็นสีแดง']
        district_type['307'] = d_type['307 | Proactive-Fiber broken']
        district_type['371'] = d_type['371 | Proactive - Fiber degrade']

        district_all['Total TT 066'] = district_type.groupby(['SCAB_DISTRICT_E'])[['066']].sum()
        district_all['Total TT 275'] = district_type.groupby(['SCAB_DISTRICT_E'])[['275']].sum()
        district_all['Total TT 307'] = district_type.groupby(['SCAB_DISTRICT_E'])[['307']].sum()
        district_all['Total TT 371'] = district_type.groupby(['SCAB_DISTRICT_E'])[['371']].sum()

        district_all['Total TT Closed'] = district_all['Total TT Closed'].replace(np.nan, 0)
        district_all['Total TT 30 Day'] = district_all['Total TT 30 Day'].replace(np.nan, 0)

        d2_percent = round(district_all.iloc[:,2] / district_all.iloc[:,1] *100, 2)
        district_all['%RR30Day'] = d2_percent
        district_all['%RR30Day'] = district_all['%RR30Day'].replace(np.nan, 0)
        

        ########## FILTER TO FIND RR [All Type]#############

        d2_rnum = district_table[(district_table.RR_ALL_FAULT_IS_RR30DAY == 'RR') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
        d2_rnum.reset_index(drop=True, inplace=True)

        ######################################################
        ########################### RR#1 #########################

        all_value_counts1 = d2_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 1]
        all_df_value_counts1 = pd.DataFrame(all_value_counts1)
        all_df_value_counts1.columns = ['RR#1']
        all_df_value_counts1.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
        district_all['RR#1'] = all_df_value_counts1.groupby('SCAB_DISTRICT_E')['RR#1'].count()

        ##########################################################

        ########################### RR#2 #########################

        all_value_counts2 = d2_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 2]
        all_df_value_counts2 = pd.DataFrame(all_value_counts2)
        all_df_value_counts2.columns = ['RR#2']
        all_df_value_counts2.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
        district_all['RR#2'] = all_df_value_counts2.groupby('SCAB_DISTRICT_E')['RR#2'].count()

        ##########################################################

        ########################### RR#3 #########################

        all_value_counts3 = d2_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 3]
        all_df_value_counts3 = pd.DataFrame(all_value_counts3)
        all_df_value_counts3.columns = ['RR#3']
        all_df_value_counts3.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
        district_all['RR#3'] = all_df_value_counts3.groupby('SCAB_DISTRICT_E')['RR#3'].count()

        ##########################################################

        ########################### RR>=4 #########################

        all_value_counts4 = d2_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x >= 4]
        all_df_value_counts4 = pd.DataFrame(all_value_counts4)
        all_df_value_counts4.columns = ['RR>=4']
        all_df_value_counts4.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
        district_all['RR>=4'] = all_df_value_counts4.groupby('SCAB_DISTRICT_E')['RR>=4'].count()

        ##########################################################

        district_all['RR#1'] = district_all['RR#1'].replace(np.nan, 0)
        district_all['RR#2'] = district_all['RR#2'].replace(np.nan, 0)
        district_all['RR#3'] = district_all['RR#3'].replace(np.nan, 0)
        district_all['RR>=4'] = district_all['RR>=4'].replace(np.nan, 0)

        district_all.loc['Grand Total']= district_all.sum(numeric_only=True, axis=0)
        d22_percent = round(district_all.iloc[-1,2] / district_all.iloc[-1,1] *100, 2)
        district_all.iloc[-1,7] = d22_percent

        st.write('ALL TYPE Accumulated:')
        st.write('#Noted : Click at the top Right Botton to view in Full Screen')
        district_all

        ###################################################################################
