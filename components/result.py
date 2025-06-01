import streamlit as st
import pandas as pd
import plotly.express as px
import time

def display_results(results, model_name):
    """Display prediction results with visualizations."""
    if not results:
        st.error("❌ No prediction results available!")
        return

    st.header("📊 Prediction Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    st.subheader("📈 Prediction Distribution")
    prediction_counts = results_df['Prediction'].value_counts().reset_index()
    prediction_counts.columns = ['Prediction', 'Count']
    color_map = {'Good': '#28a745', 'Poor': '#dc3545', 'Error - Prediction Failed': '#ffc107', 'Error - File Prep': '#6c757d'}
    fig = px.bar(prediction_counts, x='Prediction', y='Count', color='Prediction', color_discrete_map=color_map,
                title="Prediction Outcome Distribution", labels={'Count': 'Number of Patients', 'Prediction': 'Outcome'})
    fig.update_layout(xaxis_title="Predicted Outcome", yaxis_title="Number of Patients")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Detailed Patient Results")
    for _, row in results_df.iterrows():
        patient_id, prediction = row['Patient ID'], row['Prediction']
        if prediction == 'Good':
            st.markdown(f'<div class="prediction-result good-result">👤 {patient_id}: {prediction}</div>', unsafe_allow_html=True)
        elif prediction == 'Poor':
            st.markdown(f'<div class="prediction-result poor-result">👤 {patient_id}: {prediction}</div>', unsafe_allow_html=True)
        else:
            st.error(f"👤 {patient_id}: {prediction}")

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Patients", len(results))
    with col_stat2:
        st.metric("Good Outcomes", sum(1 for r in results if r['Prediction'] == 'Good'))
    with col_stat3:
        st.metric("Poor Outcomes", sum(1 for r in results if r['Prediction'] == 'Poor'))
    with col_stat4:
        st.metric("Errors", sum(1 for r in results if 'Error' in r['Prediction']))

    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv_data,
        file_name=f"eeg_predictions_{model_name.replace('/','_').replace(' ','_')}_{int(time.time())}.csv",
        mime="text/csv"
    )