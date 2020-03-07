import argparse
import json
import os.path
import random
import re
import time

import cv2
import numpy as np
import requests
from PIL import Image

from config import config


class GermanLicensePlateImagesGenerator:
    def __init__(self, output):
        self.output = output
        self.COUNTY_MARKS = np.asarray([d['CM'] for d in json.loads(open(config.GERMAN_COUNTY_MARKS, encoding='utf-8').read())])
        self.LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ"
        self.DIGITS = "0123456789"
        self.COUNTIES = ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL', 'SN', 'ST', 'SH', 'TH']
        self.MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        self.YEARS = ['06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']

        random.seed()

    @staticmethod
    def get_image_url(license_number, country, month, year):
        license_number = license_number.replace("-", "%3A").replace("Ä", "%C4").replace("Ö", "%D6").replace("Ü", "%DC")
        return "http://nummernschild.heisnbrg.net/fe/task?action=startTask&kennzeichen={0}&kennzeichenZeile2=&engschrift=false&pixelHoehe=32&breiteInMM=520&breiteInMMFest=true&sonder=FE&dd=01&mm=01&yy=00&kreis={1}&kreisName=&humm={2}&huyy={3}&sonderKreis=LEER&mm1=01&mm2=01&farbe=SCHWARZ&effekt=KEIN&tgaDownload=false".format(
            license_number, country, month, year)

    def __generate_license_number(self):
        country = random.choice(self.COUNTY_MARKS)

        letter_count = random.randint(1, 2)
        letters = "{}".format(random.choice(self.LETTERS)) if letter_count == 1 else "{}{}".format(
            random.choice(self.LETTERS), random.choice(self.LETTERS))

        min = 1 if letter_count == 2 else 1
        digit_count = random.randint(min, max((8 - len(country) - letter_count), 4))
        digits = ""
        for i in range(digit_count):
            digits += random.choice(self.DIGITS)

        return "{}-{}{}".format(country, letters, digits)

    def __create_license_plate_picture(self, n, license_number, country, front):
        file_path = self.output + '/{0}#{1}.png'.format("F" if front else "R", license_number)
        if os.path.exists(file_path):
            return False

        month = random.choice(self.MONTHS) if front else ''
        year = random.choice(self.YEARS) if front else ''

        create_image_url = GermanLicensePlateImagesGenerator.get_image_url(license_number, country, month, year)
        r = requests.get(create_image_url)
        if r.status_code != 200:
            return False

        id = re.compile('<id>(.*?)</id>', re.DOTALL | re.IGNORECASE).findall(
            r.content.decode("utf-8"))[0]

        status_url = 'http://nummernschild.heisnbrg.net/fe/task?action=status&id=%s' % id
        time.sleep(.200)
        r = requests.get(status_url)
        if r.status_code != 200:
            return False

        show_image_url = 'http://nummernschild.heisnbrg.net/fe/task?action=showInPage&id=%s'
        show_image_url = show_image_url % id
        time.sleep(.200)
        r = requests.get(show_image_url)
        if r.status_code != 200:
            return False

        # sometimes the web service returns a corrupted image, check the image by getting the size and skip if corrupted
        try:
            numpyarray = np.fromstring(r.content, np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            im = Image.fromarray(image)  # don't use cv2.imwrite() because there is a bug with utf-8 encoded filepaths
            im.save(file_path)
            print("{0:06d} : {1}".format(n, file_path))
            return True
        except:
            return False

    def generate(self, items):
        for n in range(items):
            while True:
                license_number = self.__generate_license_number()

                country = random.choice(self.COUNTIES)
                if not self.__create_license_plate_picture(n, license_number, country, True):
                    break

                time.sleep(.200)
                self.__create_license_plate_picture(n, license_number, country, False)
                time.sleep(.200)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--items", default="60000", help="Number of items to generate")
ap.add_argument("-o", "--output", default=config.PLATE_IMAGES, help="Output path")
args = vars(ap.parse_args())

lpdg = GermanLicensePlateImagesGenerator(os.path.abspath(args["output"]))
lpdg.generate(int(args["items"]))
