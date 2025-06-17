/* code for gas sensor */
#include <Blynk.h>

#define BLYNK_PRINT Serial
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>

#define BLYNK_TEMPLATE_ID "TMPL3I_BX0hNs"
#define BLYNK_TEMPLATE_NAME "Agriculture Robot"

char auth[] = "IxAL36i1k0LafHEnyTvnVipyinaKn1jb";  
char ssid[] = "Epik wifi";
char pass[] = "stonkstonks";  

float sensorValue;
#define sensorPin A0

void setup() {

  Serial.begin(9600);
  Blynk.begin(auth, ssid, pass);
}

void loop() {

 sensorValue = analogRead(sensorPin); // read analog input pin 0
  
  Serial.print("Sensor Value: ");
  Serial.print(sensorValue);

  Blynk.virtualWrite(V4, sensorValue);
  Blynk.run();  

  delay(2000); // wait 2s for next reading
}
