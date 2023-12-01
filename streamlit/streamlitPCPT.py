import streamlit as st
from copy import deepcopy

#def streamlitPCPT():
from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing, DEFAULT_CONE_PROPERTIES
from groundhog.general.soilprofile import SoilProfile, profile_from_dataframe

st.set_page_config(page_title="Groundhog PCPT processing",layout='wide')
cols = st.columns([1, 3])

with cols[0]:

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

    columns_expander = st.expander(label="Column overrides")
    with columns_expander:
        Z_COLUMN = st.text_input("Depth column override (default 'z [m]')", None)

        QC_COLUMN = st.text_input("Cone resistance column override (default 'qc [MPa]')", None)
        QC_MULTIPLIER = st.number_input("Cone resistance multiplier", value=1., step=1.,format="%.3f")

        FS_COLUMN = st.text_input("Sleeve friction column override (default 'fs [MPa]')", None)
        FS_MULTIPLIER = st.number_input("Sleeve friction multiplier", value=1., step=1.,format="%.3f")

        U2_COLUMN = st.text_input("Pore pressure column override  (default 'u2 [MPa]')", None)
        U2_MULTIPLIER = st.number_input("Pore pressure multiplier", value=1., step=1.,format="%.3f")

        PUSH_COLUMN = st.text_input("Push column override  (default 'Push')", 'Push')

    st.write(USER_FILE)

    AREA_RATIO = st.number_input("Cone area ratio", value=0.8, step=0.01,format="%.2f")
    UNIT_WEIGHT = st.number_input("Soil total unit weight [kN/m3]", value=18.0, step=0.1,format="%.1f")
    WATER_LEVEL = st.number_input("Water level below mudline [m]", value=0.0, step=0.1,format="%.1f")

    def return_None(key):
        if key == 'None' or key is None:
            return None
        else:
            return key

    with cols[1]:
        if FILE_FORMAT == file_format_options[0]:  # Excel
            if USER_FILE is not None:

                pcpt_excel = PCPTProcessing(title="PCPT Excel")
                try:
                    pcpt_excel.load_excel(
                        USER_FILE,
                        z_key=return_None(Z_COLUMN),
                        qc_key=return_None(QC_COLUMN),
                        qc_multiplier=QC_MULTIPLIER,
                        fs_key=return_None(FS_COLUMN),
                        fs_multiplier=FS_MULTIPLIER,
                        u2_key=return_None(U2_COLUMN),
                        u2_multiplier=U2_MULTIPLIER,
                        push_key=return_None(PUSH_COLUMN))

                    st.markdown("## First Few Lines of the Data")
                    st.dataframe(pcpt_excel.data)

                    st.markdown("## Visualisation of the Data")

                    cpt_chart = pcpt_excel.plot_raw_pcpt(return_fig=True)
                    cpt_chart['layout']['xaxis1'].update(title='qc [MPa]')
                    cpt_chart['layout']['xaxis2'].update(title='fs [MPa]')
                    cpt_chart['layout']['xaxis3'].update(title='u2 [MPa]')
                    cpt_chart['layout']['yaxis1'].update(title='z [m]')

                    st.plotly_chart(cpt_chart)
                except Exception as err:
                    st.markdown("Error during data loading, check column names and try again (%s)" % str(err))

                try:
                    cone_props = profile_from_dataframe(deepcopy(DEFAULT_CONE_PROPERTIES))
                    DEFAULT_CONE_PROPERTIES.loc[0, "Depth to [m]"] = pcpt_excel.data['z [m]'].max()
                    DEFAULT_CONE_PROPERTIES.loc[0, "area ratio [-]"] = AREA_RATIO
                    layering = SoilProfile({
                        'Depth from [m]': [0, ],
                        'Depth to [m]': [pcpt_excel.data['z [m]'].max(), ],
                        'Total unit weight [kN/m3]': [UNIT_WEIGHT, ],
                        'Soil type': ['Unspecified', ]
                    })
                    pcpt_excel.map_properties(
                        layer_profile=layering,
                        cone_profile=cone_props,
                        waterlevel=WATER_LEVEL
                    )
                    pcpt_excel.normalise_pcpt()
                    normalised_chart = pcpt_excel.plot_normalised_pcpt(return_fig=True)
                    normalised_chart['layout']['xaxis1'].update(title='Qt [-]')
                    normalised_chart['layout']['xaxis2'].update(title='Fr [%]')
                    normalised_chart['layout']['xaxis3'].update(title='Bq [-]')
                    normalised_chart['layout']['yaxis1'].update(title='z [m]')

                    st.markdown("## Normalised CPT data")

                    st.plotly_chart(normalised_chart)
                except Exception as err:
                    st.markdown("Error during CPT processing, check input parameters and try again (%s)" % str(err))

        else:
            st.error("Other file formats are not supported yet.")
