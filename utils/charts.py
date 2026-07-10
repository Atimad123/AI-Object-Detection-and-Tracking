"""
=========================================================
charts.py
---------------------------------------------------------
Plotly charts for AI Object Detection Dashboard

Author : Atimad BEL CAID
=========================================================
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class DashboardCharts:
    """
    Create charts for Streamlit dashboard.
    """

    # =====================================================
    # BAR CHART
    # =====================================================

    @staticmethod
    def class_bar_chart(class_counts):

        if len(class_counts) == 0:
            return go.Figure()

        df = pd.DataFrame({

            "Class": list(class_counts.keys()),

            "Count": list(class_counts.values())

        })

        fig = px.bar(

            df,

            x="Class",

            y="Count",

            color="Class",

            text="Count",

            title="Detected Objects per Class"

        )

        fig.update_layout(

            template="plotly_dark",

            height=450,

            xaxis_title="Class",

            yaxis_title="Objects",

            showlegend=False

        )

        return fig

    # =====================================================
    # PIE CHART
    # =====================================================

    @staticmethod
    def class_pie_chart(class_counts):

        if len(class_counts) == 0:
            return go.Figure()

        fig = px.pie(

            values=list(class_counts.values()),

            names=list(class_counts.keys()),

            hole=0.45,

            title="Object Distribution"

        )

        fig.update_layout(

            template="plotly_dark",

            height=450

        )

        return fig

    # =====================================================
    # CONFIDENCE HISTOGRAM
    # =====================================================

    @staticmethod
    def confidence_chart(detections):

        if len(detections) == 0:
            return go.Figure()

        confidence = [

            det["confidence"] * 100

            for det in detections

        ]

        fig = px.histogram(

            x=confidence,

            nbins=15,

            title="Confidence Distribution"

        )

        fig.update_layout(

            template="plotly_dark",

            height=450,

            xaxis_title="Confidence (%)",

            yaxis_title="Frequency"

        )

        return fig

    # =====================================================
    # FPS LINE CHART
    # =====================================================

    @staticmethod
    def fps_chart(fps_history):

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                y=fps_history,

                mode="lines+markers",

                line=dict(width=3),

                name="FPS"

            )

        )

        fig.update_layout(

            template="plotly_dark",

            title="FPS Evolution",

            xaxis_title="Frame",

            yaxis_title="FPS",

            height=400

        )

        return fig

    # =====================================================
    # SUMMARY TABLE
    # =====================================================

    @staticmethod
    def summary_table(class_counts):

        if len(class_counts) == 0:

            return pd.DataFrame()

        df = pd.DataFrame({

            "Class": list(class_counts.keys()),

            "Objects": list(class_counts.values())

        })

        df = df.sort_values(

            by="Objects",

            ascending=False

        )

        return df

    # =====================================================
    # METRICS CARDS
    # =====================================================

    @staticmethod
    def metrics(total_objects,
                fps,
                inference,
                confidence):

        return {

            "Objects": total_objects,

            "FPS": round(fps, 2),

            "Inference (ms)": round(inference, 2),

            "Confidence (%)": round(confidence, 2)

        }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    sample = {

        "Person": 8,

        "Car": 5,

        "Bus": 2,

        "Truck": 1,

        "Bicycle": 3

    }

    charts = DashboardCharts()

    print(charts.summary_table(sample))

    print(

        charts.metrics(

            total_objects=19,

            fps=31.45,

            inference=18.7,

            confidence=95.3

        )

    )