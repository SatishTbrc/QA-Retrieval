import streamlit as st
import pandas as pd
import psycopg2
import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import html

# Setting up LangChain for rephrasing text
api_key = os.getenv("API_KEY")  # Ensure your API key is correctly set in your environment variables
davinci = OpenAI(api_key=api_key, model_name="gpt-3.5-turbo-instruct")
template = """You are an experienced business analyst skilled in summarizing complex research findings into clear, concise abstracts. Generate a summary of the content from a detailed business research report. The output should be succinct with bullet points and should distill the essence of the content, highlighting key insights. All start each point with capital letter only, this is very important:

Content: {content}

Summary:"""
prompt_template = PromptTemplate(template=template, input_variables=["content"])
llm_chain = LLMChain(prompt=prompt_template, llm=davinci)

# Database connection configuration
#conn_str = "host=localhost port=5432 dbname=AI_tool user=postgres password=Postgre@273."
conn_str = os.environ.get("DATABASE_URL")

# Define the function to rephrase content using LangChain
def rephrase_with_langchain(content):
    try:
        rephrased_content = llm_chain.run(content=content)
        return rephrased_content
    except Exception as e:
        st.error(f"Failed to generate rephrased content: {str(e)}")
        return None

def save_to_database(selected_market, selected_data_type, rephrased_content, conn_str):
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            # Check if the entry already exists
            check_query = """
                SELECT COUNT(*) FROM output_data WHERE "Market" = LOWER(%s) AND "Data" = LOWER(%s)
            """
            cursor.execute(check_query, (selected_market, selected_data_type))
            exists = cursor.fetchone()[0]

            if exists == 0:  # If the entry does not exist, insert new data
                insert_query = """
                    INSERT INTO output_data ("Market", "Data", "Answer") VALUES (LOWER(%s), LOWER(%s), %s)
                """
                cursor.execute(insert_query, (selected_market, selected_data_type, rephrased_content))
                conn.commit()
                #st.success("Data successfully saved to the database.")
            
def fetch_from_output(selected_market, selected_data_type, conn_str):
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            # Check if the entry exists in the database
            entry_exists_query = """
                SELECT COUNT(*) FROM output_data WHERE "Market" = LOWER(%s) AND "Data" = LOWER(%s)
            """
            cursor.execute(entry_exists_query, (selected_market, selected_data_type))
            entry_exists = cursor.fetchone()[0]

            if entry_exists > 0:  # If the entry exists, retrieve and display it
                fetch_answer_query = """
                    SELECT "Answer" FROM output_data WHERE "Market" = LOWER(%s) AND "Data" = LOWER(%s)
                """
                cursor.execute(fetch_answer_query, (selected_market, selected_data_type))
                rephrased_content = cursor.fetchone()[0]
                return rephrased_content
    return None  # Return None if no data is found

def get_available_market_size(market, conn_str):
    available_market_size = []
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                CASE
                    WHEN COUNT(DISTINCT geography) > 1 THEN TRUE
                    WHEN COUNT(DISTINCT geography) = 1 AND MAX(geography) IS NOT NULL THEN TRUE
                    ELSE FALSE
                END
            FROM
                market_data
            WHERE
                segment = %s;
            """, (market,))
        row = cursor.fetchone()
        if row:
            available_market_size.append("Market Size")
    except (Exception, psycopg2.Error) as error:
        print("Error fetching data types:", error)
    finally:
        if conn:
            cursor.close()
            conn.close()
    return available_market_size


def get_hyperlink(selected_market, selected_country, conn_str):
    query = """
        SELECT "Hyperlink" from public.market_data WHERE LOWER(segment) = LOWER(%s) AND LOWER(geography) = LOWER(%s)
    """

    hyperlink = None
    try:
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (selected_market,selected_country))
                row = cursor.fetchone()
                if row:
                    hyperlink = row[0]
    except (Exception, psycopg2.Error) as error:
        print("Error fetching hyperlink:", error)
    return hyperlink

def get_reportlink(selected_market, conn_str):
    query = """
        SELECT "Reportlink" from public.market_data WHERE LOWER(segment) = LOWER(%s)
    """

    hyperlink = None
    try:
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (selected_market,))
                row = cursor.fetchone()
                if row:
                    hyperlink = row[0]
    except (Exception, psycopg2.Error) as error:
        print("Error fetching hyperlink:", error)
    return hyperlink

def get_available_data_types(market, conn_str):
    available_data_types = []
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                CASE WHEN "Market Trends" IS NOT NULL THEN 'Market Trends' ELSE NULL END AS "Market Trends",
                CASE WHEN "Market Drivers" IS NOT NULL THEN 'Market Drivers' ELSE NULL END AS "Market Drivers",
                CASE WHEN "Market Restraints" IS NOT NULL THEN 'Market Restraints' ELSE NULL END AS "Market Restraints",
                CASE WHEN "Competitive Landscape" IS NOT NULL THEN 'Competitive Landscape' ELSE NULL END AS "Competitive Landscape"
            FROM 
                public.market_data 
            WHERE 
                LOWER(segment) = LOWER(%s)
                AND (
                    "Market Trends" IS NOT NULL 
                    OR "Market Drivers" IS NOT NULL 
                    OR "Market Restraints" IS NOT NULL 
                    OR "Competitive Landscape" IS NOT NULL
                );
            """, (market,))
        row = cursor.fetchone()
        if row:
            available_data_types = [col for col in row if col is not None]
    except (Exception, psycopg2.Error) as error:
        print("Error fetching data types:", error)
    finally:
        if conn:
            cursor.close()
            conn.close()
    return available_data_types

# Database functions
def check_market_in_database(market_name, conn_str):
    query = """
        SELECT segment FROM public.market_data WHERE LOWER(segment) = LOWER(%s)
    """
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (market_name,))
            row = cursor.fetchone()
            if row:
                return True
            else:
                return False

def check_data_availability(selected_market, selected_data_type, conn_str):
    query = """
        SELECT "{}" FROM public.market_data WHERE LOWER(segment) = LOWER(%s)
    """.format(selected_data_type)
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (selected_market,))
            row = cursor.fetchone()
            return row


def check_global_data_availability(selected_market, conn_str):
    query = """
        SELECT DISTINCT geography FROM public.market_data WHERE LOWER(segment) = LOWER(%s)
    """
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (selected_market,))
            geographies = cursor.fetchall()

            # If only one geography and it's marked as 'Global'
            if len(geographies) == 1 and geographies[0][0].lower() == 'global':
                return 'Global'
            # If multiple geographies or single non-Global geography
            elif len(geographies) > 1 or (len(geographies) == 1 and geographies[0][0].lower() != 'global'):
                return True
            else:
                return False


def check_region_data_availability(selected_market, selected_country, conn_str):
    query = """
        SELECT "Market Size" FROM public.market_data 
        WHERE LOWER(segment) = LOWER(%s) 
        AND LOWER(geography) = LOWER(%s)
    """
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (selected_market, selected_country))
            row = cursor.fetchone()
            if row is not None:
                return True
            else:
                return False

def get_top_5_similar_markets_from_database(selected_market, conn_str):
    similar_markets = []
    query_starts_with = """
        SELECT DISTINCT segment 
        FROM public.market_data 
        WHERE LOWER(segment) LIKE LOWER(%s)
        LIMIT 5;
    """
    
    query_contains = """
        SELECT DISTINCT segment 
        FROM public.market_data 
        WHERE LOWER(segment) LIKE LOWER(%s)
        AND LOWER(segment) NOT LIKE LOWER(%s)
        LIMIT 5;
    """
    
    try:
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cursor:
                # First query: segments starting with the selected_market
                cursor.execute(query_starts_with, (selected_market + '%',))
                rows = cursor.fetchall()
                similar_markets.extend([row[0] for row in rows])
                
                # Check if we need more suggestions to fill the top 5
                if len(similar_markets) < 5:
                    # Second query: segments containing the selected_market
                    cursor.execute(query_contains, ('%' + selected_market + '%', selected_market + '%'))
                    rows = cursor.fetchall()
                    similar_markets.extend([row[0] for row in rows])
                    
                # Ensure we only return the top 5 suggestions
                similar_markets = similar_markets[:5]
                
    except Exception as e:
        print(f"Database error: {e}")
    
    return similar_markets
def find_region_for_country(country_name, conn_str):
    query = """
        SELECT region
        FROM public.region_country
        WHERE LOWER(country) = LOWER(%s);
    """
    region = None
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (country_name,))
            result = cursor.fetchone()
            if result:
                region = result[0]
    return region

def get_top_5_geographies_for_market_and_region(selected_market, region, conn_str):
    geographies = []
    query = """
        SELECT DISTINCT md.geography
        FROM public.market_data AS md
        JOIN public.region_country AS rc ON md.geography = rc.country
        WHERE LOWER(md.segment) = LOWER(%s)
        AND rc.region = %s
        AND md.geography IS NOT NULL
        ORDER BY md.geography
        LIMIT 5;
    """
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (selected_market, region))
            rows = cursor.fetchall()
            geographies = [row[0] for row in rows]
    return geographies


def fetch_answer_from_database(selected_market, data_type, selected_country, conn_str):
    years = {
        "Historical data": ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        "Forecast data": ['2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033']
    }
    
    query = ""
    if data_type in years:
        year_clause = ", ".join(f"y{year}" for year in years[data_type])
        query = f"""
            SELECT {year_clause} from public.market_data 
            WHERE LOWER(segment) = LOWER(%s) AND LOWER(geography) = LOWER(%s)
        """
    else:
        return "Invalid data type", None
    
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (selected_market, selected_country))
            row = cursor.fetchone()
            if row:
                # Fetching row values from the database and formatting them to two decimal places
                formatted_values = ["{:.2f}".format(float(value)) if value is not None else None for value in row]
                
                # Constructing the row values
                row_values = [selected_country, selected_market, "Sales", "Fixed USD", "Billion"] + formatted_values
                
                # Constructing the headers
                headers = ["Geography", "Segment", "Type", "Value", "Units"] + years[data_type]
                
                # Transposing the table
                data = [headers, row_values]
                
                # Return the transposed table
                return data, None
            else:
                return None, "No data available"



# Streamlit app functions
def handle_selected_market(selected_market):
    if check_market_in_database(selected_market, conn_str):
        return True
    else:
        st.write("Unfortunately, we don’t cover this exact market in the Global Market Model, but here are some similar markets you might be interested in:")
        #similar_markets_query = f"SELECT segment FROM (SELECT DISTINCT segment FROM public.market_data WHERE LOWER(segment) LIKE LOWER('%{selected_market}%')) subquery ORDER BY RANDOM() LIMIT 5"
        similar_markets = get_top_5_similar_markets_from_database(selected_market, conn_str)
        if not similar_markets:
            st.error("We don't have this market, please enter a valid market name.")
            return False
        selected_similar_market = st.selectbox("Select a similar market:",[""] + similar_markets)
        if selected_similar_market:
            #selected_market == selected_similar_market
            # Execute the block of code for the selected similar market
            success_selected_market = handle_selected_market(selected_similar_market)
            if success_selected_market:
                selected_data_type = None
                # Check available data types for the selected market
                available_data_types = get_available_data_types(selected_similar_market, conn_str)
            
                data_type_options = ["Market Size"] + available_data_types
                selected_data_type = st.selectbox("What type of data are you looking for?", ["Select Option Below"] + data_type_options)
                if selected_data_type != "Select Option Below":
                    st.session_state.data_type = selected_data_type

                if selected_data_type in ["Market Trends", "Market Drivers", "Market Restraints", "Competitive Landscape"]:
                    # Try to fetch existing rephrased content from the database
                    rephrased_content = fetch_from_output(selected_similar_market, selected_data_type, conn_str)
                    if rephrased_content:
                        if selected_data_type == "Market Trends":
                            st.write(f"Key trends in the {selected_similar_market} market are:")
                        if selected_data_type == "Market Drivers":
                            st.write(f"Key drivers in the {selected_similar_market} market are:")
                        if selected_data_type == "Market Restraints":
                            st.write(f"Key restraints in the {selected_similar_market} market are:")
                        if selected_data_type == "Competitive Landscape":
                            st.write(f"Key insights on the competitive landscape of the {selected_similar_market} market are:")
                        st.write(rephrased_content)
                        reportlink = get_reportlink(selected_similar_market, conn_str)
                        st.write(f"If you need further details :  {reportlink}")
                        further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                        further_datatype = "select option below"
                        entere_button = st.button("Continue")
                        if further_assistance or entere_button:
                            # Clear session state variables
                            st.session_state.market = further_assistance
                            if 'data_type' not in st.session_state:
                                st.session_state.data_type = ""
                            st.session_state.country = ""

                            # Rerun the app
                            st.experimental_rerun()
                    else:
                        row = check_data_availability(selected_similar_market, selected_data_type, conn_str)
                        if row and row[0]:  # Checks that row is not None and row[0] is not an empty string or other falsy value
                            rephrased_content = rephrase_with_langchain(row[0])
                            if rephrased_content:
                                if selected_data_type == "Market Trends":
                                    st.write(f"Key trends in the {selected_similar_market} market are:")
                                if selected_data_type == "Market Drivers":
                                    st.write(f"Key drivers in the {selected_similar_market} market are:")
                                if selected_data_type == "Market Restraints":
                                    st.write(f"Key restraints in the {selected_similar_market} market are:")
                                if selected_data_type == "Competitive Landscape":
                                    st.write(f"Key insights on the competitive landscape of the {selected_similar_market} market are:")
                                st.write(rephrased_content)
                                reportlink = get_reportlink(selected_similar_market, conn_str)
                                st.write(f"If you need further details :  {reportlink}")
                                save_to_database(selected_similar_market, selected_data_type, rephrased_content, conn_str)
                                further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                further_datatype = "select option below"
                                entere_button = st.button("Continue")
                                if further_assistance or entere_button:
                                    # Clear session state variables
                                    st.session_state.market = further_assistance
                                    if 'data_type' not in st.session_state:
                                        st.session_state.data_type = ""
                                    st.session_state.country = ""

                                    # Rerun the app
                                    st.experimental_rerun()
                            else:
                                st.write("Unable to rephrase the content at this time.")
                        else:
                            st.write(f"Unfortunately, we don’t have the {selected_data_type.lower()} available for this market on the Global Market Model, but we cover the historic and forecast market size.")
                            st.write("Let's proceed with the Market Size data.")
                            selected_data_type = "Market Size"

                if selected_data_type == "Market Size":
                    data_available_at_global_level = False
                    if check_data_availability(selected_similar_market, selected_data_type, conn_str):
                        data_available_at_global_level = check_global_data_availability(selected_similar_market, conn_str)

                    if data_available_at_global_level:
                        if data_available_at_global_level == 'Global':
                            # Handle the global case
                            st.write("Data available only at a global level.")
                            historical_or_forecast = st.radio("Do you need historical data or forecasts?", ["Select option below","Historical data", "Forecast data"])
                                
                            if historical_or_forecast == "Historical data":
                                data, error = fetch_answer_from_database(selected_similar_market, "Historical data", "global", conn_str)
                                if error:
                                    st.write(error)
                                else:
                                    st.write(f"Here's the historical data of {selected_similar_market} of Global for the year 2013-2023:")
                                    df = pd.DataFrame(data[1:], columns=data[0])
                                    df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                    st.dataframe(df)
                                    hyperlink = get_hyperlink(selected_similar_market,'Global', conn_str)
                                    st.write(f"If you need further details or comparisons:  {hyperlink}")
                                    further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                    further_datatype = "select option below"
                                    entere_button = st.button("Continue")
                                    if further_assistance or entere_button:
                                        # Clear session state variables
                                        st.session_state.market = further_assistance
                                        if 'data_type' not in st.session_state:
                                            st.session_state.data_type = ""
                                        st.session_state.country = ""

                                        # Rerun the app
                                        st.experimental_rerun()

                            elif historical_or_forecast == "Forecast data":
                                data, error = fetch_answer_from_database(selected_similar_market, "Forecast data", "global", conn_str)
                                if error:
                                    st.write(error)
                                else:
                                    st.write(f"Here's the forecast data of {selected_similar_market} of Global for the year 2023-2033:")
                                    df = pd.DataFrame(data[1:], columns=data[0])
                                    df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                    st.dataframe(df)
                                    hyperlink = get_hyperlink(selected_similar_market,'Global', conn_str)
                                    st.write(f"If you need further details or comparisons:  {hyperlink}")
                                    further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                    further_datatype = "select option below"
                                    entere_button = st.button("Continue")
                                    if further_assistance or entere_button:
                                        # Clear session state variables
                                        st.session_state.market = further_assistance
                                        if 'data_type' not in st.session_state:
                                            st.session_state.data_type = ""
                                        st.session_state.country = ""

                                        # Rerun the app
                                        st.experimental_rerun()
                        elif data_available_at_global_level:
                            selected_country = st.text_input("Which geography are you interested in? Please specify a country or region:", value=st.session_state.country)  # Use st.session_state.country as the default value
                            if selected_country:
                                success_geography = process_market_size_data(selected_similar_market, selected_country, selected_data_type)
                                if success_geography:
                                    # Continue with historical_or_forecast radio button and answer retrieval
       
                                    historical_or_forecast = st.radio("Do you need historical data or forecasts?", ["Select option below","Historical data", "Forecast data"])
                                
                                    if historical_or_forecast == "Historical data":
                                        data, error = fetch_answer_from_database(selected_similar_market, "Historical data", selected_country, conn_str)
                                        if error:
                                            st.write(error)
                                        else:
                                            st.write(f"Here's the historical data of {selected_similar_market} of {selected_country} for the year 2013 - 2023:")
                                            df = pd.DataFrame(data[1:], columns=data[0])
                                            df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                            st.dataframe(df)
                                            hyperlink = get_hyperlink(selected_similar_market,selected_country, conn_str)
                                            st.write(f"If you need further details or comparisons:  {hyperlink}")
                                            further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                            further_datatype = "select option below"
                                            entere_button = st.button("Continue")
                                            if further_assistance or entere_button:
                                                # Clear session state variables
                                                st.session_state.market = further_assistance
                                                if 'data_type' not in st.session_state:
                                                    st.session_state.data_type = ""
                                                st.session_state.country = ""

                                                # Rerun the app
                                                st.experimental_rerun()

                                    elif historical_or_forecast == "Forecast data":
                                        data, error = fetch_answer_from_database(selected_similar_market, "Forecast data", selected_country, conn_str)
                                        if error:
                                            st.write(error)
                                        else:
                                            st.write(f"Here's the forecast data of {selected_similar_market} of {selected_country} for the year 2023 - 2033:")
                                            df = pd.DataFrame(data[1:], columns=data[0])
                                            df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                            st.dataframe(df)
                                            hyperlink = get_hyperlink(selected_similar_market,selected_country, conn_str)
                                            st.write(f"If you need further details or comparisons:  {hyperlink}")
                                            further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                            further_datatype = "select option below"
                                            entere_button = st.button("Continue")
                                            if further_assistance or entere_button:
                                                # Clear session state variables
                                                st.session_state.market = further_assistance
                                                if 'data_type' not in st.session_state:
                                                    st.session_state.data_type = ""
                                                st.session_state.country = ""

                                                # Rerun the app
                                                st.experimental_rerun()



def process_market_size_data(selected_market, selected_country, selected_data_type):
    market_size_available = check_data_availability(selected_market, selected_data_type, conn_str)

    if market_size_available:
        region_data_available = check_region_data_availability(selected_market, selected_country, conn_str)

        if region_data_available:
            st.write(f"Market Size data for {selected_country} in the {selected_market} market is available.")
            return True  # Indicate success
        else:
            st.write(f"Unfortunately, we don’t cover this geography in the Global Market Model, but here are some of the similar geographies you might be interested in:")
            region = find_region_for_country(selected_country, conn_str)
            if region:
                similar_geographies = get_top_5_geographies_for_market_and_region(selected_market, region, conn_str)
                
                similar_geographies = [""] + similar_geographies
                selected_similar_geography = st.selectbox("Select a similar geography:", similar_geographies, key=f"{selected_market}_geo")

                # Check if a geography has been selected
                if selected_similar_geography:
                    # Store the selected geography in session state to preserve across reruns
                    #st.session_state.selected_similar_geography = selected_similar_geography

                    success_geography = process_market_size_data(selected_market, selected_similar_geography, selected_data_type)
                    if success_geography:
                        historical_or_forecast = st.radio("Do you need historical data or forecasts?", ["Select option below","Historical data", "Forecast data"])
                                
                        if historical_or_forecast == "Historical data":
                            data, error = fetch_answer_from_database(selected_market, "Historical data", selected_similar_geography, conn_str)
                            if error:
                                st.write(error)
                            else:
                                st.write(f"Here's the historical data of {selected_market} of {selected_similar_geography} for the year 2013 - 2023:")
                                df = pd.DataFrame(data[1:], columns=data[0])
                                df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                st.dataframe(df)
                                hyperlink = get_hyperlink(selected_market,selected_similar_geography, conn_str)
                                st.write(f"If you need further details or comparisons:  {hyperlink}")
                                further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                further_datatype = "select option below"
                                entere_button = st.button("Continue")
                                if further_assistance or entere_button:
                                    # Clear session state variables
                                    st.session_state.market = further_assistance
                                    if 'data_type' not in st.session_state:
                                        st.session_state.data_type = ""
                                    st.session_state.country = ""

                                    # Rerun the app
                                    st.experimental_rerun()

                        elif historical_or_forecast == "Forecast data":
                            data, error = fetch_answer_from_database(selected_market, "Forecast data", selected_similar_geography, conn_str)
                            if error:
                                st.write(error)
                            else:
                                st.write(f"Here's the forecast data of {selected_market} of {selected_similar_geography} for the year 2023-2033:")
                                df = pd.DataFrame(data[1:], columns=data[0])
                                df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                st.dataframe(df)
                                hyperlink = get_hyperlink(selected_market,selected_similar_geography, conn_str)
                                st.write(f"If you need further details or comparisons:  {hyperlink}")
                                further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                further_datatype = "select option below"
                                entere_button = st.button("Continue")
                                if further_assistance or entere_button:
                                    # Clear session state variables
                                    st.session_state.market = further_assistance
                                    if 'data_type' not in st.session_state:
                                        st.session_state.data_type = ""
                                    st.session_state.country = ""

                                    # Rerun the app
                                    st.experimental_rerun()
            else:
                st.error("Please enter a valid geography.")
                return False
       
                            
    return False  # Indicate failure

def main():
    
    if 'market' not in st.session_state:
        st.session_state.market = ""

    if 'data_type' not in st.session_state:
        st.session_state.data_type = ""

    if 'country' not in st.session_state:  # Add this line to initialize the 'country' variable
        st.session_state.country = ""

    # Reset selected_data_type if further assistance is provided
    if 'selected_data_type' not in st.session_state:
        st.session_state.selected_data_type = "select option below"
    elif st.session_state.get('reset_selected_data_type', False):
        st.session_state.selected_data_type = "select option below"
        st.session_state.reset_selected_data_type = False


    selected_market = st.text_input("", value = st.session_state.market)
    enter_button = st.button("Enter")

    if selected_market or enter_button:
        

        success_selected_market = handle_selected_market(selected_market)
        if success_selected_market:
            selected_data_type = None
            # Check available data types for the selected market
            available_data_types = get_available_data_types(selected_market, conn_str)
            available_market_size = get_available_market_size(selected_market, conn_str)
            #data_type_options = []

            # Only add "Market Size" if it's available
            #if is_market_size_available(selected_market, conn_str):
                #data_type_options.append("Market Size")

            data_type_options = available_market_size + available_data_types
            selected_data_type = st.selectbox("What type of data are you looking for?", ["Select Option Below"] + data_type_options)
            if selected_data_type != "Select Option Below":
                st.session_state.data_type = selected_data_type

                if selected_data_type in ["Market Trends", "Market Drivers", "Market Restraints", "Competitive Landscape"]:
                    # Try to fetch existing rephrased content from the database
                    rephrased_content = fetch_from_output(selected_market, selected_data_type, conn_str)
                    if rephrased_content:
                        if selected_data_type == "Market Trends":
                            st.write(f"Key trends in the {selected_market} market are:")
                        if selected_data_type == "Market Drivers":
                            st.write(f"Key drivers in the {selected_market} market are:")
                        if selected_data_type == "Market Restraints":
                            st.write(f"Key restraints in the {selected_market} market are:")
                        if selected_data_type == "Competitive Landscape":
                            st.write(f"Key insights on the competitive landscape of the {selected_market} market are:")
                        #st.write(rephrased_content)
                        # Define CSS for consistent font style
                        st.markdown("""
                        <style>
                            .font-style {
                                font-family: 'Roboto', sans-serif;
                                font-size: 16px;
                            }
                        </style>
                        """, unsafe_allow_html=True)
                        # Escape the content to ensure no special formatting is retained
                        escaped_content = html.escape(rephrased_content)

                        # Display the rephrased content within a <div> using the .font-style class
                        st.markdown(f'<div class="font-style">{escaped_content}</div>', unsafe_allow_html=True)
                        reportlink = get_reportlink(selected_market, conn_str)
                        st.write(f"If you need further details :  {reportlink}")
                        further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                        further_datatype = "select option below"
                        entere_button = st.button("Continue")
                        if further_assistance or entere_button:
                            st.session_state.market = further_assistance
                            st.session_state.data_type = ""
                            st.session_state.reset_selected_data_type = True  # Trigger reset on next run
                            # Rerun the app
                            st.experimental_rerun()
                    else:
                        row = check_data_availability(selected_market, selected_data_type, conn_str)
                        if row and row[0]:
                            rephrased_content = rephrase_with_langchain(row[0])
                            if rephrased_content:
                                if selected_data_type == "Market Trends":
                                    st.write(f"Key trends in the {selected_market} market are:")
                                if selected_data_type == "Market Drivers":
                                    st.write(f"Key drivers in the {selected_market} market are:")
                                if selected_data_type == "Market Restraints":
                                    st.write(f"Key restraints in the {selected_market} market are:")
                                if selected_data_type == "Competitive Landscape":
                                    st.write(f"Key insights on the competitive landscape of the {selected_market} market are:")
                                #st.write(rephrased_content)
                                # Define CSS for consistent font style
                                st.markdown("""
                                <style>
                                    .font-style {
                                        font-family: 'Roboto', sans-serif;
                                        font-size: 16px;
                                    }
                                </style>
                                """, unsafe_allow_html=True)
                                # Escape the content to ensure no special formatting is retained
                                escaped_content = html.escape(rephrased_content)

                                # Display the rephrased content within a <div> using the .font-style class
                                st.markdown(f'<div class="font-style">{escaped_content}</div>', unsafe_allow_html=True)
                                reportlink = get_reportlink(selected_market, conn_str)
                                st.write(f"If you need further details :  {reportlink}")
                                save_to_database(selected_market, selected_data_type, rephrased_content, conn_str)
                                further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                further_datatype = "select option below"
                                entere_button = st.button("Continue")
                                if further_assistance or entere_button:
                                    st.session_state.market = further_assistance
                                    st.session_state.data_type = ""
                                    st.session_state.reset_selected_data_type = True  # Trigger reset on next run
                                    # Rerun the app
                                    st.experimental_rerun()
                            else:
                                st.write("Unable to rephrase the content at this time.")
                        else:
                            st.write(f"Unfortunately, we don’t have the {selected_data_type.lower()} available for this market on the Global Market Model, but we cover the historic and forecast market size.")
                            st.write("Let's proceed with the Market Size data.")
                            selected_data_type = "Market Size"


            if selected_data_type == "Market Size":
                data_available_at_global_level = False
                if check_data_availability(selected_market, selected_data_type, conn_str):
                    data_available_at_global_level = check_global_data_availability(selected_market, conn_str)

                if data_available_at_global_level:
                    if data_available_at_global_level == 'Global':
                        # Handle the global case
                        st.write("Data available only at a global level.")
                        historical_or_forecast = st.radio("Do you need historical data or forecasts?", ["Select option below","Historical data", "Forecast data"])
                                
                        if historical_or_forecast == "Historical data":
                            data, error = fetch_answer_from_database(selected_market, "Historical data", "global", conn_str)
                            if error:
                                st.write(error)
                            else:
                                st.write(f"Here's the historical data of {selected_market} of Global for the year 2013-2023:")
                                df = pd.DataFrame(data[1:], columns=data[0])
                                df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                st.dataframe(df)
                                hyperlink = get_hyperlink(selected_market,"global", conn_str)
                                st.write(f"If you need further details or comparisons:  {hyperlink}")
                                further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                further_datatype = "select option below"
                                entere_button = st.button("Continue")
                                if further_assistance or entere_button:
                                    # Clear session state variables
                                    st.session_state.market = further_assistance
                                    if 'data_type' not in st.session_state:
                                        st.session_state.data_type = ""
                                    st.session_state.country = ""

                                    # Rerun the app
                                    st.experimental_rerun()
                        elif historical_or_forecast == "Forecast data":
                            data, error = fetch_answer_from_database(selected_market, "Forecast data", "global", conn_str)
                            if error:
                                st.write(error)
                            else:
                                st.write(f"Here's the forecast data of {selected_market} of Global for the year 2023-2033:")
                                df = pd.DataFrame(data[1:], columns=data[0])
                                df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                st.dataframe(df)
                                hyperlink = get_hyperlink(selected_market,"global", conn_str)
                                st.write(f"If you need further details or comparisons:  {hyperlink}")
                                further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                further_datatype = "select option below"
                                entere_button = st.button("Continue")
                                if further_assistance or entere_button:
                                    # Clear session state variables
                                    st.session_state.market = further_assistance
                                    if 'data_type' not in st.session_state:
                                        st.session_state.data_type = ""
                                    st.session_state.country = ""

                                    # Rerun the app
                                    st.experimental_rerun()
                    elif data_available_at_global_level:
                        selected_country = st.text_input("Which geography are you interested in? Please specify a country or region:", value=st.session_state.country)  # Use st.session_state.country as the default value
                        if selected_country:
                            success_geography = process_market_size_data(selected_market, selected_country, selected_data_type)
                            if success_geography:
                                # Continue with historical_or_forecast radio button and answer retrieval
       
                                historical_or_forecast = st.radio("Do you need historical data or forecasts?", ["Select option below","Historical data", "Forecast data"])
                                
                                if historical_or_forecast == "Historical data":
                                    data, error = fetch_answer_from_database(selected_market, "Historical data", selected_country, conn_str)
                                    if error:
                                        st.write(error)
                                    else:
                                        st.write(f"Here's the historical data of {selected_market}  of {selected_country} for the year 2013-2023:")
                                        df = pd.DataFrame(data[1:], columns=data[0])
                                        df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                        st.dataframe(df)
                                        hyperlink = get_hyperlink(selected_market,selected_country, conn_str)
                                        st.write(f"If you need further details or comparisons:  {hyperlink}")
                                        further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                        further_datatype = "select option below"
                                        entere_button = st.button("Continue")
                                        if further_assistance or entere_button:
                                            # Clear session state variables
                                            st.session_state.market = further_assistance
                                            if 'data_type' not in st.session_state:
                                                st.session_state.data_type = ""
                                            st.session_state.country = ""

                                            # Rerun the app
                                            st.experimental_rerun()
                                elif historical_or_forecast == "Forecast data":
                                    data, error = fetch_answer_from_database(selected_market, "Forecast data", selected_country, conn_str)
                                    if error:
                                        st.write(error)
                                    else:
                                        st.write(f"Here's the forecast data of {selected_market} of {selected_country} for the year 2023-2033:")
                                        df = pd.DataFrame(data[1:], columns=data[0])
                                        df = df.set_index(df.columns[0], drop=True)  # Set the index to None, removing it
                                        st.dataframe(df)
                                        hyperlink = get_hyperlink(selected_market,selected_country, conn_str)
                                        st.write(f"If you need further details or comparisons:  {hyperlink}")
                                        further_assistance = st.text_input("What would you like to search for next? Please specify which market you are seeking information on in the text box below ?")
                                        further_datatype = "select option below"
                                        entere_button = st.button("Continue")
                                        if further_assistance or entere_button:
                                            # Clear session state variables
                                            st.session_state.market = further_assistance
                                            if 'data_type' not in st.session_state:
                                                st.session_state.data_type = ""
                                            st.session_state.country = ""

                                            # Rerun the app
                                            st.experimental_rerun()


if __name__ == "__main__":
    st.set_page_config("Question Answering App", layout="wide")
    hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    logo_path = "logo-TBRC.png"
    st.image(logo_path, width=200)
    st.title("Global Market Model AI-bot")
    st.write("Helping you find market information")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Hello! Please specify which market you are seeking information on? You can type the market name in the text box below:")

    main()

