# How to run: Open the cmd, cd to the folder of the Python file. "streamlit run streamlitShallowFoundation.py"

import streamlit as st

import sys
sys.path.append("../groundhog")

def streamlitShallowFoundation():
    from groundhog.shallowfoundations.capacity import (
        ShallowFoundationCapacityDrained,
        ShallowFoundationCapacityUndrained,
    )


    st.title("Groundhog Bearing Capacity Calculations")
    cols = st.columns([1, 1, 1])

    with cols[0]:
        FOUNDATION_GEOMETRY = st.selectbox(
            "Foundation geometry", options=["Rectangle", "Circular"]
        )

        if FOUNDATION_GEOMETRY == "Rectangle":
            LENGTH = st.number_input("Length of the foundation (m)", value=8.00)
            WIDTH = st.number_input("Width of the foundation (m)", value=5.00)
        elif FOUNDATION_GEOMETRY == "Circular":
            DIAMETER = st.number_input("Diameter of the foundation", value=10.0)
        else:
            st.write("Error on foundation geometry.")

        DEPTH = st.number_input("Depth of foundation (m)", value=0.00)

        DRAINAGE = st.selectbox(
            "Drainage type of the foundation", options=["Drained", "Undrained"]
        )

        SKIRTED = st.checkbox("Skirted Foundation?", value=False)


    if DRAINAGE == "Undrained":
        with cols[1]:
            UNIT_WEIGHT = st.number_input("Unit weight of the soil (kN/m3)", value=16.0)
            SU_BASE = st.number_input("Undrained shear strength at base (kPa)", value=10.0)
            SU_INCREASE = st.number_input(
                "Increase of undrained shear strength with depth (kPa/m)", value=0.0
            )

            ECCENTRICITY_WIDTH = st.number_input(
                "Eccentricity of load along the width (m)", value=0.0
            )
            ECCENTRICITY_LENGTH = st.number_input(
                "Eccentricity of load along the length (m)", value=0.0
            )

            undrained_centric = ShallowFoundationCapacityUndrained(
                title="Undrained capacity for centric loading"
            )

            if FOUNDATION_GEOMETRY == "Rectangle":
                undrained_centric.set_geometry(length=LENGTH, width=WIDTH)
            elif FOUNDATION_GEOMETRY == "Circular":
                undrained_centric.set_geometry(diameter=DIAMETER)

            undrained_centric.set_soilparameters_undrained(
                unit_weight=UNIT_WEIGHT, su_base=SU_BASE, su_increase=SU_INCREASE
            )

            undrained_centric.set_eccentricity(
                eccentricity_width=ECCENTRICITY_WIDTH,
                eccentricity_length=ECCENTRICITY_LENGTH,
            )

            undrained_centric.calculate_bearing_capacity()
            NET_BEARING_PRESSURE = undrained_centric.net_bearing_pressure
            CAPACITY = undrained_centric.capacity

            undrained_centric.calculate_sliding_capacity()
            SLIDING_FULL = undrained_centric.sliding_full

            undrained_centric.calculate_envelope()

            undrained_centric_plot = undrained_centric.plot_envelope(showfig=False)
        with cols[-1]:
            st.plotly_chart(undrained_centric_plot)

    elif DRAINAGE == "Drained":

        with cols[1]:
            drained_centric = ShallowFoundationCapacityDrained(
                title="Drained capacity for centric loading"
            )

            if FOUNDATION_GEOMETRY == "Rectangle":
                drained_centric.set_geometry(length=LENGTH, width=WIDTH)
            elif FOUNDATION_GEOMETRY == "Circular":
                drained_centric.set_geometry(diameter=DIAMETER)

            UNIT_WEIGHT = st.number_input(
                "Effective unit weight of the soil (kN/m3)", value=9.0
            )
            FRICTION_ANGLE = st.number_input("Friction angle of the soil (deg)", value=38.0)
            EFFECTIVE_STRESS_BASE = st.number_input(
                "Effective stress at the base of the foundation (kPa)", value=0.0
            )

            drained_centric.set_soilparameters_drained(
                effective_unit_weight=UNIT_WEIGHT,
                friction_angle=FRICTION_ANGLE,
                effective_stress_base=EFFECTIVE_STRESS_BASE,
            )

            ECCENTRICITY_WIDTH = st.number_input(
                "Eccentricity of load along the width (m)", value=0.0
            )
            ECCENTRICITY_LENGTH = st.number_input(
                "Eccentricity of load along the length (m)", value=0.0
            )

            drained_centric.set_eccentricity(
                eccentricity_length=ECCENTRICITY_LENGTH,
                eccentricity_width=ECCENTRICITY_WIDTH,
            )

            drained_centric.calculate_bearing_capacity()

            drained_centric.calculate_sliding_capacity(vertical_load=1000)

            drained_centric.calculate_envelope()

            drained_centric_plot = drained_centric.plot_envelope(showfig=False)

        with cols[-1]:
            st.plotly_chart(drained_centric_plot)
