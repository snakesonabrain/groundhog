import streamlit as st
from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing


file_format_options = ["Excel", ".asc", "AGS"]
FILE_FORMAT = st.selectbox(
    "Select the file format for PCPT data.",
    options=file_format_options,
    help="If the selected and uploaded file format does not match, the program will fail.",
)

USER_FILE = st.file_uploader(
    "Go ahead.",
    accept_multiple_files=False,
    help="Upload the file format selected above. Currently, only one file is accepted.",
)

st.write(USER_FILE)

if FILE_FORMAT == file_format_options[0]:  # Excel
    if USER_FILE is not None:

        pcpt_excel = PCPTProcessing(title="PCPT Excel")
        pcpt_excel.load_excel(USER_FILE, u2_key="u [kPa]", u2_multiplier=0.001)

        st.markdown("## First Few Lines of the Data")
        st.write(pcpt_excel.data.head())

        st.markdown("## Visualisation of the Data")

        st.plotly_chart(pcpt_excel.plot_raw_pcpt(return_fig=True))


else:
    st.error("Other file formats are not supported yet.")
