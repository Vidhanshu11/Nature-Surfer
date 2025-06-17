#define BLYNK_TEMPLATE_ID "TMPL3I_BX0hNs"
#define BLYNK_TEMPLATE_NAME "Agriculture Robot"

#include <Wire.h>
#include <TinyGPS++.h>
#include <SoftwareSerial.h>
#include <BlynkSimpleEsp8266.h>

#define GPS_TX_PIN 4  
#define GPS_RX_PIN 5  

SoftwareSerial gpsSerial(GPS_TX_PIN, GPS_RX_PIN);

TinyGPSPlus gps;

char auth[] = "IxAL36i1k0LafHEnyTvnVipyinaKn1jb";

BlynkTimer timer;

void setup()
{
  Serial.begin(9600);
  Blynk.begin(auth, "Epik wifi", "stonkstonks");
  gpsSerial.begin(9600);
  timer.setInterval(100L, updateGPSData); // Send GPS data to the map every 1 second
}

void loop()
{
  Blynk.run();
  timer.run();
}

void updateGPSData()
{
  while (gpsSerial.available() > 0)
  {
    if (gps.encode(gpsSerial.read()))
    {
      float lat = gps.location.lat();
      float lon = gps.location.lng();
      Blynk.virtualWrite(V8, lat);
      Blynk.virtualWrite(V9, lon);

      Serial.print("Latitude: ");
      Serial.println(lat, 6);
      Serial.print("Longitude: ");
      Serial.println(lon, 6);

      Blynk.virtualWrite(V7, lon, lat); // Send the coordinates to the map widget
    }
  }
}
