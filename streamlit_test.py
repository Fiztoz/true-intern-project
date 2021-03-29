import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt

import base64
from io import BytesIO



st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title='True Data Visualize',  # String or None. Strings get appended with "• Streamlit". 
	page_icon='./t1.png',  # String, anything supported by st.image, or None.
)


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df,file_name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)
    button_id = 'button'
    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{file_name}.xlsx" id="{button_id}" href="data:application/octet-stream;base64,{b64.decode()}">Download : {file_name}</a><br></br>'
    )
   
    return dl_link



st.image('./true.png')


st.sidebar.header('User Input')

dataview1 = st.sidebar.selectbox('Data View Option', ['RR_Fault','District_Fault_Table'])


up_file = st.file_uploader('Upload your data file', type=["txt","xlsx","csv"])
if up_file is not None:
    st.write('You selected `%s`' % up_file.name)
else:
    st.write('You selected `%s`' % up_file)


if up_file is not None:
    df = pd.read_table(up_file)


    if dataview1 == 'RR_Fault':

        province =st.sidebar.selectbox('Province Option',df.SCAB_PROVINCE_E.unique())

        c1, c2,c3= st.beta_columns((1, 1,3))
        with c1:
            rr_select = st.selectbox('View Option', ['Month','Week'])
        with c2:
            if rr_select == 'Week':
                wk_value = st.selectbox('Choose Week',df.FAULT_COMPLETE_WEEK.unique())
                data_select = 'RR_Accumulated'
                data_select2 = 'RR_Accumulated'

        if rr_select == 'Month':
            allticket = df[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','FAULT_FAULT_TYPE_CD','FAULT_DEPARTMENT_GROUP','FAULT_DEPARTMENT','RR_TRUCK_FAULT_IS_RR30DAY','RR_ALL_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE','FAULT_COMPLETE_WEEK']]
            allticket = allticket[allticket.SCAB_PROVINCE_E == province]
        else:
            allticket = df[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','FAULT_FAULT_TYPE_CD','FAULT_DEPARTMENT_GROUP','FAULT_DEPARTMENT','RR_TRUCK_FAULT_IS_RR30DAY','RR_ALL_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE','FAULT_COMPLETE_WEEK']]
            allticket = allticket[allticket.SCAB_PROVINCE_E == province]
            allticket = allticket[allticket.FAULT_COMPLETE_WEEK == wk_value]

        ##########      RR Truck Roll      ###########

        
        rr= df
        rr['RR_TRUCK_FAULT_IS_RR30DAY'] = rr['RR_TRUCK_FAULT_IS_RR30DAY'].replace(np.nan, 0)
        r30 = pd.get_dummies(rr.RR_TRUCK_FAULT_IS_RR30DAY)
        rr['RR'] = r30.RR
        if rr_select == 'Month':
            nrt = rr[(rr.SCAB_PROVINCE_E == province) & (rr.FAULT_DEPARTMENT_GROUP == 'Truck Roll') & (rr.FAULT_TICKET_TYPE == 'Regular Fault')]
        else:
            nrt = rr[(rr.SCAB_PROVINCE_E == province) & (rr.FAULT_DEPARTMENT_GROUP == 'Truck Roll') & (rr.FAULT_TICKET_TYPE == 'Regular Fault')]
            nrt = nrt[nrt.FAULT_COMPLETE_WEEK == wk_value]

        nrt.reset_index(drop=True, inplace=True)
        st.write("Truck Roll Data :")
        nrt[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','FAULT_DEPARTMENT_GROUP','RR_TRUCK_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE','FAULT_COMPLETE_WEEK']]
        st.markdown(get_table_download_link(nrt,f"Truck_Data_{province}({rr_select})"), unsafe_allow_html=True)

        if rr_select == 'Month':
            c1, c2,c3= st.beta_columns((1, 1,3))
            with c1:
              data_select = st.selectbox('View Data[Truck Roll]', ['RR_Accumulated','FAULT_WEEK'])

        if data_select == 'RR_Accumulated':
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
            # new_rr['RR Rate Accumulate(%)'] = new_rr['RR Rate Accumulate(%)'].astype(str)
            st.write('RR Truck Roll Report : ')
            tran_rr = new_rr.transpose()
            # tran_rr =tran_rr.astype(str)
            # tran_rr
            changetype = tran_rr.transpose()
            changetype['RR Rate Accumulate(%)'] = changetype['RR Rate Accumulate(%)'].astype(str)
            changetype.iloc[:,0:5] = changetype.iloc[:,0:5].astype(int)
            # asdw.dtypes 
            tran_rr2 = changetype.transpose()
            tran_rr2

            st.markdown(get_table_download_link(tran_rr2,f"RR_Truck_Report_{province}"), unsafe_allow_html=True)
        
            chart_new_rr = new_rr
            chart_new_rr.reset_index(drop=False, inplace=True)

            Y=chart_new_rr['RR Rate Accumulate(%)']
            X=chart_new_rr.index

            fig2 = px.line(chart_new_rr, y='RR Rate Accumulate(%)', x='FAULT_COMPLETE_DATE',text ='RR Rate Accumulate(%)',title="Truck Roll RR Rate(%)" )
            fig2.update_traces(texttemplate='%{text:.2f}%', textposition='top center')
            st.plotly_chart(fig2, use_container_width=True)

        if data_select == 'FAULT_WEEK':
            fault_week = allticket.groupby(['FAULT_COMPLETE_WEEK'])[['FAULT_TICKET_NUMBER']].count()
            fault_week.rename(columns={'FAULT_TICKET_NUMBER':'All Ticket Count'},inplace = True)

            fault_week['FAULT_TICKET_NUMBER'] = nrt.groupby(['FAULT_COMPLETE_WEEK'])[['FAULT_TICKET_NUMBER']].count()
            fault_week['FAULT_TICKET_NUMBER'] = fault_week['FAULT_TICKET_NUMBER'].replace(np.nan, 0)
            fault_rate = round(fault_week.iloc[:,1] / fault_week.iloc[:,0] *100, 2)
            fault_week['Fault_Rate(%)'] = fault_rate
            fault_week['RR'] = nrt.groupby(['FAULT_COMPLETE_WEEK'])[['RR']].sum()
            fault_week['RR'] = fault_week['RR'].replace(np.nan, 0)
            fault_rr_rate = round(fault_week.iloc[:,3] / fault_week.iloc[:,1] *100, 2)
            fault_week['Fault_RR_Rate(%)'] = fault_rr_rate
            st.write('Truck Roll Report : ')
            fault_week1 = fault_week.transpose()
            fault_week1 = fault_week1.transpose()
            fault_week1['Fault_Rate(%)'] = fault_week1['Fault_Rate(%)'].astype(str) 
            fault_week1['Fault_RR_Rate(%)'] = fault_week1['Fault_RR_Rate(%)'].astype(str) 
            fault_week1
            st.markdown(get_table_download_link(fault_week1,f"Truck_{data_select}_{province}"), unsafe_allow_html=True)

            chart_fault_week = fault_week
            chart_fault_week.reset_index(drop=False, inplace=True)

            X= chart_fault_week['FAULT_COMPLETE_WEEK']
            Y = chart_fault_week['Fault_RR_Rate(%)']
            Z = chart_fault_week['Fault_Rate(%)']
            fig = go.Figure(data=go.Scatter(name='RR Rate',x = X, y = Y))
            fig.add_trace(go.Scatter(name='Fault Rate', x=X, y=Z))
            for x,y in zip(X,Y):
                a = '%s' %(y)
                a = a+' %'
                fig.add_annotation(x=x, y=y,
                    text=a,
                    showarrow=False,
                    yshift=10)
            for x,y in zip(X,Z):
                a = '%s' %(y)
                a = a+' %'
                fig.add_annotation(x=x, y=y,
                    text=a,
                    showarrow=False,
                    yshift=10)
            fig.update_layout(showlegend=True)
            fig.update_layout(title="Truck Roll Rate in Week(%)", xaxis_title = 'Week', yaxis_title = 'Rate(%)')
            st.plotly_chart(fig, use_container_width=True)
        #############################################

        ##########      RR All Fault      ###########


        allrr = df
        allrr['RR_ALL_FAULT_IS_RR30DAY'] = allrr['RR_ALL_FAULT_IS_RR30DAY'].replace(np.nan, 0)
        allr30 = pd.get_dummies(allrr.RR_ALL_FAULT_IS_RR30DAY)
        allrr['RR'] = allr30.RR

        if rr_select == 'Month':
            all_nrt = allrr[(allrr.SCAB_PROVINCE_E == province) & (allrr.FAULT_TICKET_TYPE == 'Regular Fault')]
        else:
            all_nrt = allrr[(allrr.SCAB_PROVINCE_E == province) & (allrr.FAULT_TICKET_TYPE == 'Regular Fault')]
            all_nrt = all_nrt[all_nrt.FAULT_COMPLETE_WEEK == wk_value]

        all_nrt.reset_index(drop=True, inplace=True)
        st.write("All Type Data :")
        all_nrt[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','RR_ALL_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE','FAULT_COMPLETE_WEEK']]
        st.markdown(get_table_download_link(all_nrt,f"ALL_Data_{province}({rr_select})"), unsafe_allow_html=True)
        if rr_select == 'Month':
            c1, c2,c3= st.beta_columns((1, 1,3))
            with c1:
              data_select2 = st.selectbox('View Data[ALL Type]', ['RR_Accumulated','FAULT_WEEK'])

        if data_select2 == 'RR_Accumulated':
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
            
            all_changetype = all_tran_rr.transpose()
            all_changetype['RR Rate Accumulate(%)'] = all_changetype['RR Rate Accumulate(%)'].astype(str)
            all_changetype.iloc[:,0:5] = all_changetype.iloc[:,0:5].astype(int)
            all_tran_rr2 = all_changetype.transpose()
            all_tran_rr2
            st.markdown(get_table_download_link(all_tran_rr2,f"RR_ALL_Report_{province}"), unsafe_allow_html=True)
                
            all_chart_new_rr = all_new_rr
            all_chart_new_rr.reset_index(drop=False, inplace=True)


            all_Y=all_chart_new_rr['RR Rate Accumulate(%)']
            all_X=all_chart_new_rr.index


            all_fig2 = px.line(all_chart_new_rr, y='RR Rate Accumulate(%)', x='FAULT_COMPLETE_DATE',text ='RR Rate Accumulate(%)',title="All Type RR Rate(%)" )
            all_fig2.update_traces(texttemplate='%{text:.2f}%', textposition='top center')
            st.plotly_chart(all_fig2, use_container_width=True)
        
        if data_select2 == 'FAULT_WEEK':
            all_fault_week = allticket.groupby(['FAULT_COMPLETE_WEEK'])[['FAULT_TICKET_NUMBER']].count()
            all_fault_week.rename(columns={'FAULT_TICKET_NUMBER':'All Ticket Count'},inplace = True)

            all_fault_week['FAULT_TICKET_NUMBER'] = all_nrt.groupby(['FAULT_COMPLETE_WEEK'])[['FAULT_TICKET_NUMBER']].count()
            all_fault_week['FAULT_TICKET_NUMBER'] = all_fault_week['FAULT_TICKET_NUMBER'].replace(np.nan, 0)
            all_fault_rate = round(all_fault_week.iloc[:,1] / all_fault_week.iloc[:,0] *100, 2)
            all_fault_week['Fault_Rate(%)'] = all_fault_rate
            all_fault_week['RR'] = all_nrt.groupby(['FAULT_COMPLETE_WEEK'])[['RR']].sum()
            all_fault_week['RR'] = all_fault_week['RR'].replace(np.nan, 0)
            all_fault_rr_rate = round(all_fault_week.iloc[:,3] / all_fault_week.iloc[:,1] *100, 2)
            all_fault_week['Fault_RR_Rate(%)'] = all_fault_rr_rate
            st.write('Truck Roll Report : ')
            all_fault_week1 = all_fault_week.transpose()
            all_fault_week1 = all_fault_week1.transpose()
            all_fault_week1['Fault_Rate(%)'] = all_fault_week1['Fault_Rate(%)'].astype(str) 
            all_fault_week1['Fault_RR_Rate(%)'] = all_fault_week1['Fault_RR_Rate(%)'].astype(str) 
            all_fault_week1
            st.markdown(get_table_download_link(all_fault_week1,f"ALL_{data_select}_{province}"), unsafe_allow_html=True)

            chart_all_fault_week = all_fault_week
            chart_all_fault_week.reset_index(drop=False, inplace=True)

            all_X= chart_all_fault_week['FAULT_COMPLETE_WEEK']
            all_Y = chart_all_fault_week['Fault_RR_Rate(%)']
            all_Z = chart_all_fault_week['Fault_Rate(%)']
            all_fig = go.Figure(data=go.Scatter(name='RR Rate',x = all_X, y = all_Y))
            all_fig.add_trace(go.Scatter(name='Fault Rate', x=all_X, y=all_Z))
            for x,y in zip(all_X,all_Y):
                a = '%s' %(y)
                a = a+' %'
                all_fig.add_annotation(x=x, y=y,
                    text=a,
                    showarrow=False,
                    yshift=10)
            for x,y in zip(all_X,all_Z):
                a = '%s' %(y)
                a = a+' %'
                all_fig.add_annotation(x=x, y=y,
                    text=a,
                    showarrow=False,
                    yshift=10)
            all_fig.update_layout(showlegend=True)
            all_fig.update_layout(title="All Type Rate in Week(%)", xaxis_title = 'Week', yaxis_title = 'Rate(%)')
            st.plotly_chart(all_fig, use_container_width=True)

        #############################################

    if dataview1 == 'District_Fault_Table':

        province =st.sidebar.selectbox('Province Option',df.SCAB_PROVINCE_E.unique())

        district_table = df[['FAULT_TICKET_NUMBER','FAULT_TELEPHONE_NUMBER','FAULT_TICKET_TYPE','FAULT_FAULT_TYPE_CD','FAULT_DEPARTMENT_GROUP','FAULT_DEPARTMENT','RR_TRUCK_FAULT_IS_RR30DAY','RR_ALL_FAULT_IS_RR30DAY','SCAB_PROVINCE_E','SCAB_DISTRICT_E','FAULT_COMPLETE_DATE','FAULT_COMPLETE_WEEK']]
        

        c1, c2,c3= st.beta_columns((1, 1,3))
        with c1:
            d_select = st.selectbox('View Option', ['Month','Week','Day'])
        with c2:
            if d_select == 'Week':
                wk_dvalue = st.selectbox('Choose Week',district_table.FAULT_COMPLETE_WEEK.unique())
            elif d_select == 'Day':
                d_value = st.selectbox('Choose Day',sorted(district_table.FAULT_COMPLETE_DATE.unique()))      
        
        district_table = district_table[district_table.SCAB_PROVINCE_E == province]
        district_table['RR_TRUCK_FAULT_IS_RR30DAY'] = district_table['RR_TRUCK_FAULT_IS_RR30DAY'].replace(np.nan, 0)
        district_table['RR_ALL_FAULT_IS_RR30DAY'] = district_table['RR_ALL_FAULT_IS_RR30DAY'].replace(np.nan, 0)
        district_table.reset_index(drop=True, inplace=True)


        ############### TRUCK ROLL ##################
        if d_select =='Month':
            d1 = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
            d1rr = pd.get_dummies(d1.RR_TRUCK_FAULT_IS_RR30DAY)
            d1['RR_Truck'] = d1rr.RR
        elif d_select == 'Week':
            d1 = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]   
            d1rr = pd.get_dummies(d1.RR_TRUCK_FAULT_IS_RR30DAY)
            d1['RR_Truck'] = d1rr.RR
            d1 = d1[d1.FAULT_COMPLETE_WEEK == wk_dvalue]
        else :
            d1 = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]  
            d1rr = pd.get_dummies(d1.RR_TRUCK_FAULT_IS_RR30DAY)
            d1['RR_Truck'] = d1rr.RR
            d1 = d1[d1.FAULT_COMPLETE_DATE == d_value]
        d1.reset_index(drop=True, inplace=True)

        district = district_table.groupby(['SCAB_DISTRICT_E'])[['FAULT_TICKET_TYPE']].count()
        district.rename(columns={'FAULT_TICKET_TYPE':'All Ticket Count'},inplace = True)
        district['Total TT Closed'] = d1.groupby(['SCAB_DISTRICT_E'])[['RR_TRUCK_FAULT_IS_RR30DAY']].count()



        district['Total TT 30 Day'] = d1.groupby(['SCAB_DISTRICT_E'])[['RR_Truck']].sum()


        if d_select =='Month':
            district_type = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll')]
        elif d_select == 'Week':
            district_type = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll')]
            district_type = district_type[district_type.FAULT_COMPLETE_WEEK == wk_dvalue]
        else :
            district_type = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll')]
            district_type = district_type[district_type.FAULT_COMPLETE_DATE == d_value]

        district_type.reset_index(drop=True, inplace=True)

        d_type = pd.get_dummies(district_type.FAULT_FAULT_TYPE_CD)
        num = 0
        district_type['066'] = num
        district_type['275'] = num
        district_type['307'] = num
        district_type['371'] = num 

        if '066 | Proactive' in d_type.columns:
            district_type['066'] = d_type['066 | Proactive']
        
        if '275 | ไฟ Los ติด เป็นสีแดง' in d_type.columns:
            district_type['275'] = d_type['275 | ไฟ Los ติด เป็นสีแดง']
        
        if '307 | Proactive-Fiber broken' in d_type.columns:
            district_type['307'] = d_type['307 | Proactive-Fiber broken']
        
        if '371 | Proactive - Fiber degrade' in d_type.columns:
            district_type['371'] = d_type['371 | Proactive - Fiber degrade']


        district['Total TT 066'] = district_type.groupby(['SCAB_DISTRICT_E'])[['066']].sum()
        district['Total TT 275'] = district_type.groupby(['SCAB_DISTRICT_E'])[['275']].sum()
        district['Total TT 307'] = district_type.groupby(['SCAB_DISTRICT_E'])[['307']].sum()
        district['Total TT 371'] = district_type.groupby(['SCAB_DISTRICT_E'])[['371']].sum()

        district['Total TT Closed'] = district['Total TT Closed'].replace(np.nan, 0)
        district['Total TT 30 Day'] = district['Total TT 30 Day'].replace(np.nan, 0)
        district['Total TT 066'] = district['Total TT 066'].replace(np.nan, 0)
        district['Total TT 275'] = district['Total TT 275'].replace(np.nan, 0)
        district['Total TT 307'] = district['Total TT 307'].replace(np.nan, 0)
        district['Total TT 371'] = district['Total TT 371'].replace(np.nan, 0)

        d_percent = round(district.iloc[:,2] / district.iloc[:,1] *100, 2)
        district['%RR30Day'] = d_percent
        district['%RR30Day'] = district['%RR30Day'].replace(np.nan, 0)
        district['%RR30Day'] = district['%RR30Day'].astype(str)
        

        ########## FILTER TO FIND RR [TRUCK ROLL]#############
        if d_select =='Month':
            d1_rnum = district_table[(district_table.RR_TRUCK_FAULT_IS_RR30DAY == 'RR') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
        elif d_select == 'Week':
            d1_rnum = district_table[(district_table.RR_TRUCK_FAULT_IS_RR30DAY == 'RR') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
            d1_rnum = d1_rnum[d1_rnum.FAULT_COMPLETE_WEEK == wk_dvalue]
        else :
            d1_rnum = district_table[(district_table.RR_TRUCK_FAULT_IS_RR30DAY == 'RR') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
            d1_rnum = d1_rnum[d1_rnum.FAULT_COMPLETE_DATE == d_value]

        
        # d1_rnum = district_table[(district_table.RR_TRUCK_FAULT_IS_RR30DAY == 'RR') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
        d1_rnum.reset_index(drop=True, inplace=True)
        
        ######################################################
        ########################### RR#1 #########################

        if d1_rnum.empty:
            district['RR#1'] = num
        else:
            value_counts1 = d1_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 1]
            df_value_counts1 = pd.DataFrame(value_counts1)
            df_value_counts1.columns = ['RR#1']
            df_value_counts1.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
            district['RR#1'] = df_value_counts1.groupby('SCAB_DISTRICT_E')['RR#1'].count()

        ##########################################################

        ########################### RR#2 #########################

        if d1_rnum.empty:
            district['RR#2'] = num
        else:
            value_counts2 = d1_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 2]
            df_value_counts2 = pd.DataFrame(value_counts2)
            df_value_counts2.columns = ['RR#2']
            df_value_counts2.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
            district['RR#2'] = df_value_counts2.groupby('SCAB_DISTRICT_E')['RR#2'].count()

        ##########################################################

        ########################### RR#3 #########################

        if d1_rnum.empty:
            district['RR#3'] = num
        else:
            value_counts3 = d1_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 3]
            df_value_counts3 = pd.DataFrame(value_counts3)
            df_value_counts3.columns = ['RR#3']
            df_value_counts3.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
            district['RR#3'] = df_value_counts3.groupby('SCAB_DISTRICT_E')['RR#3'].count()

        ##########################################################

        ########################### RR>=4 #########################

        if d1_rnum.empty:
            district['RR>=4'] = num
        else:
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

        st.write('Truck Roll :')
        st.write('#Noted : Click at the top Right Botton to view in Full Screen')
        district
        st.markdown(get_table_download_link(district,f'RR30Day Truck Roll_{province}_{d_select}'), unsafe_allow_html=True)

        ###################################################################################

        #################### ALL FAULT #########################
        if d_select =='Month':
            d2 = district_table[(district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
            d2rr = pd.get_dummies(d2.RR_ALL_FAULT_IS_RR30DAY)
            d2['All_Truck'] = d2rr.RR
        elif d_select == 'Week':
            d2 = district_table[(district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
            d2rr = pd.get_dummies(d2.RR_ALL_FAULT_IS_RR30DAY)
            d2['All_Truck'] = d2rr.RR
            d2 = d2[d2.FAULT_COMPLETE_WEEK == wk_dvalue]
        else :
            d2 = district_table[(district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
            d2rr = pd.get_dummies(d2.RR_ALL_FAULT_IS_RR30DAY)
            d2['All_Truck'] = d2rr.RR
            d2 = d2[d2.FAULT_COMPLETE_DATE == d_value]

        d2.reset_index(drop=True, inplace=True)

        district_all = district_table.groupby(['SCAB_DISTRICT_E'])[['FAULT_TICKET_TYPE']].count()
        district_all.rename(columns={'FAULT_TICKET_TYPE':'All Tiket Count'},inplace = True)
        district_all['Total TT Closed'] = d2.groupby(['SCAB_DISTRICT_E'])[['RR_ALL_FAULT_IS_RR30DAY']].count()



        district_all['Total TT 30 Day'] = d2.groupby(['SCAB_DISTRICT_E'])[['All_Truck']].sum()

        if d_select =='Month':
            district_type = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll')]
        elif d_select == 'Week':
            district_type = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll')]
            district_type = district_type[district_type.FAULT_COMPLETE_WEEK == wk_dvalue]
        else :
            district_type = district_table[(district_table.FAULT_DEPARTMENT_GROUP == 'Truck Roll')]
            district_type = district_type[district_type.FAULT_COMPLETE_DATE == d_value]

        district_type.reset_index(drop=True, inplace=True)

        d_type = pd.get_dummies(district_type.FAULT_FAULT_TYPE_CD)

        num = 0
        district_type['066'] = num
        district_type['275'] = num
        district_type['307'] = num
        district_type['371'] = num 

        if '066 | Proactive' in d_type.columns:
            district_type['066'] = d_type['066 | Proactive']
        
        if '275 | ไฟ Los ติด เป็นสีแดง' in d_type.columns:
            district_type['275'] = d_type['275 | ไฟ Los ติด เป็นสีแดง']
        
        if '307 | Proactive-Fiber broken' in d_type.columns:
            district_type['307'] = d_type['307 | Proactive-Fiber broken']
        
        if '371 | Proactive - Fiber degrade' in d_type.columns:
            district_type['371'] = d_type['371 | Proactive - Fiber degrade']


        district_all['Total TT 066'] = district_type.groupby(['SCAB_DISTRICT_E'])[['066']].sum()
        district_all['Total TT 275'] = district_type.groupby(['SCAB_DISTRICT_E'])[['275']].sum()
        district_all['Total TT 307'] = district_type.groupby(['SCAB_DISTRICT_E'])[['307']].sum()
        district_all['Total TT 371'] = district_type.groupby(['SCAB_DISTRICT_E'])[['371']].sum()

        district_all['Total TT Closed'] = district_all['Total TT Closed'].replace(np.nan, 0)
        district_all['Total TT 30 Day'] = district_all['Total TT 30 Day'].replace(np.nan, 0)
        district_all['Total TT 066'] = district_all['Total TT 066'].replace(np.nan, 0)
        district_all['Total TT 275'] = district_all['Total TT 275'].replace(np.nan, 0)
        district_all['Total TT 307'] = district_all['Total TT 307'].replace(np.nan, 0)
        district_all['Total TT 371'] = district_all['Total TT 371'].replace(np.nan, 0)

        d2_percent = round(district_all.iloc[:,2] / district_all.iloc[:,1] *100, 2)
        district_all['%RR30Day'] = d2_percent
        district_all['%RR30Day'] = district_all['%RR30Day'].replace(np.nan, 0)
        district_all['%RR30Day'] = district_all['%RR30Day'].astype(str)
        

        ########## FILTER TO FIND RR [All Type]#############

        if d_select =='Month':
            d2_rnum = district_table[(district_table.RR_ALL_FAULT_IS_RR30DAY == 'RR') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]

        elif d_select == 'Week':
            d2_rnum = district_table[(district_table.RR_ALL_FAULT_IS_RR30DAY == 'RR') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
            d2_rnum = d2_rnum[d2_rnum.FAULT_COMPLETE_WEEK == wk_dvalue]
        else :
            d2_rnum = district_table[(district_table.RR_ALL_FAULT_IS_RR30DAY == 'RR') & (district_table.FAULT_TICKET_TYPE == 'Regular Fault')]
            d2_rnum = d2_rnum[d2_rnum.FAULT_COMPLETE_DATE == d_value]

        d2_rnum.reset_index(drop=True, inplace=True)

        ######################################################
        ########################### RR#1 #########################

        if d2_rnum.empty:
            district_all['RR#1'] = num
        else:
            all_value_counts1 = d2_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 1]
            all_df_value_counts1 = pd.DataFrame(all_value_counts1)
            all_df_value_counts1.columns = ['RR#1']
            all_df_value_counts1.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
            district_all['RR#1'] = all_df_value_counts1.groupby('SCAB_DISTRICT_E')['RR#1'].count()

        ##########################################################

        ########################### RR#2 #########################

        if d2_rnum.empty:
            district_all['RR#2'] = num
        else:
            all_value_counts2 = d2_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 2]
            all_df_value_counts2 = pd.DataFrame(all_value_counts2)
            all_df_value_counts2.columns = ['RR#2']
            all_df_value_counts2.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
            district_all['RR#2'] = all_df_value_counts2.groupby('SCAB_DISTRICT_E')['RR#2'].count()

        ##########################################################

        ########################### RR#3 #########################

        if d2_rnum.empty:
            district_all['RR#3'] = num
        else:
            all_value_counts3 = d2_rnum.groupby('SCAB_DISTRICT_E')['FAULT_TELEPHONE_NUMBER'].value_counts().loc[lambda x : x == 3]
            all_df_value_counts3 = pd.DataFrame(all_value_counts3)
            all_df_value_counts3.columns = ['RR#3']
            all_df_value_counts3.reset_index(level=['SCAB_DISTRICT_E','FAULT_TELEPHONE_NUMBER'], inplace=True)
            district_all['RR#3'] = all_df_value_counts3.groupby('SCAB_DISTRICT_E')['RR#3'].count()

        ##########################################################

        ########################### RR>=4 #########################

        if d2_rnum.empty:
            district_all['RR>=4'] = num
        else:
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

        st.write('ALL TYPE :')
        st.write('#Noted : Click at the top Right Botton to view in Full Screen')
        district_all

        st.markdown(get_table_download_link(district_all,f'RR30Day All Type_{province}_{d_select}'), unsafe_allow_html=True)

        ###################################################################################