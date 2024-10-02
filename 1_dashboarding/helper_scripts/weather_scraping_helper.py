"""
weather_scraping_helper.py

This module contains utility functions to assist with web scraping weather data.
It includes functions to scrape weather forecasts, format the data, and handle
potential changes in the website's UI. These functions can be used in conjunction
with a web driver (e.g., Selenium) to automate data extraction tasks from weather-related websites.

Functions:
- get_8_day_forecast(driver): Scrapes an 8-day weather forecast from a given webpage.
"""

import pandas as pd
import numpy as np
from typing import Literal
from selenium.webdriver.remote.webdriver import WebDriver


def choose_and_click_temperature_unit(unit_type: Literal["celsius", "fahrenheit"], driver: WebDriver) -> None:
    """
    Selects and clicks the temperature unit on a webpage using the provided web driver.

    Args:
        unit_type (str): The type of temperature unit to select. Acceptable values are:
            - 'celsius': Selects the Celsius unit.
            - 'fahrenheit': Selects the Fahrenheit unit.
        driver (webdriver): A Selenium WebDriver instance used to interact with the webpage.

    Raises:
        ValueError: If the provided `unit_type` is not 'celsius' or 'fahrenheit'.
    """

    if unit_type == "celsius":
        # temperature celsius unit
        temp_unit_celsius_xpath = "/html/body/main/div[2]/div[2]/div/div/div[1]/div[2]/div[2]"
        driver.find_element("xpath", temp_unit_celsius_xpath).click()

    elif unit_type == "fahrenheit":
        # temperature fahrenheit unit
        temp_unit_fs_xpath = "/html/body/main/div[2]/div[2]/div/div/div[1]/div[2]/div[3]"
        driver.find_element("xpath", temp_unit_fs_xpath).click()
    else:
        raise ValueError("Invalid unit_type. Must be 'celsius' or 'fahrenheit'.")


# weather icon
def get_current_temperature(driver: WebDriver) -> str:
    """
    Scrapes and returns the current temperature from a webpage using the provided web driver.

    Args:
        driver (WebDriver): A Selenium WebDriver instance used to interact with the webpage.

    Returns:
        str: The current temperature as a string, extracted from the webpage.

    Raises:
        NoSuchElementException: If the element for the current temperature is not found on the page.
    """
    current_temp = driver.find_element("xpath", "/html/body/main/div[2]/div[3]/div[1]/div[1]/div[2]/div[1]").text
    return current_temp


def check_if_any_weather_alerts(driver: WebDriver) -> str:
    """
    Checks for weather alerts on a webpage and returns the alert text if found.

    Args:
        driver (WebDriver): A Selenium WebDriver instance used to interact with the webpage.

    Returns:
        str: The weather alert message as a string. If no alert is found or an error occurs,
        returns 'No weather Alert'.

    Raises:
        NoSuchElementException: If the weather alert element is not found. This is caught internally,
        and 'No weather Alert' is returned instead.
    """
    # weather alert
    try:
        weather_alert_xpath = "/html/body/main/div[2]/div[3]/div[1]/div[1]/div[2]/div[2]"
        weather_alert_string = driver.find_element("xpath", weather_alert_xpath).text
    except Exception:
        weather_alert_string = 'No weather Alert'
    return weather_alert_string


def get_8_day_forecast(driver: WebDriver) -> pd.DataFrame:
    """
    Scrapes an 8-day weather forecast from a website using a web driver.

    Args:
        driver (webdriver): A Selenium WebDriver instance used to interact with the webpage.

    Returns:
        pandas.DataFrame: A DataFrame containing scraped forecast data with the following columns:
            - 'day': The day number in the forecast (1 to 8).
            - 'day_date': The date or day of the forecast.
            - 'temp': The forecasted temperature.
            - 'rain_cloud_pred': The predicted rain or cloud cover.

    Raises:
        ValueError: If any of the necessary elements cannot be found (e.g., due to a UI update)
    """

    elem_list = []
    day_date_text_list = []
    temp_text_list = []
    rain_cloud_pred_text_list = []

    try:
        for elem in np.arange(1, 9):
            day_date_xpath = f"/html/body/main/div[2]/div[3]/div[2]/div[2]/ul/li[{elem}]/span"
            temp_xpath = f"/html/body/main/div[2]/div[3]/div[2]/div[2]/ul/li[{elem}]/div/div/span"
            rain_cloud_pred_xpath = f"/html/body/main/div[2]/div[3]/div[2]/div[2]/ul/li[{elem}]/div/span[1]"

            day_date_text = driver.find_element("xpath", day_date_xpath).text
            temp_text = driver.find_element("xpath", temp_xpath).text
            rain_cloud_pred_text = driver.find_element("xpath", rain_cloud_pred_xpath).text

            elem_list.append(elem)
            day_date_text_list.append(day_date_text)
            temp_text_list.append(temp_text)
            rain_cloud_pred_text_list.append(rain_cloud_pred_text)

        # Create a DataFrame with the scraped data
        forecast_df = pd.DataFrame({'day': elem_list,
                                    'day_date': day_date_text_list,
                                    'temp': temp_text_list,
                                    'rain_cloud_pred': rain_cloud_pred_text_list
                                    })
    except Exception as e:
        # Provide detailed error messages to ease debugging
        forecast_df = pd.DataFrame(columns=['day', 'day_date', 'temp', 'rain_cloud_pred'])
        raise ValueError(f"Couldn't find xpath elements. The site's UI probably got updated. Error: {e}")

    return forecast_df


def get_precipitation_information(driver: WebDriver) -> str:
    """
    Scrapes and returns precipitation information from a webpage using the provided web driver.

    Args:
        driver (WebDriver): A Selenium WebDriver instance used to interact with the webpage.

    Returns:
        str: A string containing the precipitation information. If no precipitation data is found or an error occurs,
        returns 'No precipitation information'.

    Raises:
        NoSuchElementException: If the precipitation element is not found. This is handled internally,
        and 'No precipitation information' is returned instead.
    """
    try:
        # precipitation
        prec_path = "/html/body/main/div[2]/div[3]/div[1]/div[2]/div/a/div/div"
        prec = driver.find_element("xpath", prec_path)
        prec_text = "".join(prec.text.split('\n')[:1])
    except Exception:
        prec_text = 'No precipitation information'
    return prec_text
