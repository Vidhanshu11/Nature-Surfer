/* soil moisture*/
#include <Blynk.h>

#define BLYNK_PRINT Serial
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>

#define BLYNK_TEMPLATE_ID "TMPL3I_BX0hNs"
#define BLYNK_TEMPLATE_NAME "Agriculture Robot"

char auth[] = "IxAL36i1k0LafHEnyTvnVipyinaKn1jb";  
char ssid[] = "Epik wifi"; 
char pass[] = "stonkstonks";  

#define sensorPin A0

void setup() {

Serial.begin(9600);
Blynk.begin(auth, ssid, pass);
}

void loop() {

  Serial.print("Analog output: ");
  
  delay(500);

  int sensorValue = analogRead(sensorPin);  // Read the analog value from sensor

  int outputValue = map(sensorValue, 0, 1023, 255, 0); 

  Serial.print(outputValue); 
  Blynk.virtualWrite(V3, outputValue); 
  Blynk.run();  
  
  delay(2000); // wait 2s for next reading
}
