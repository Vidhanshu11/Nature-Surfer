/* code for ph */
#include <Blynk.h>

#define BLYNK_PRINT Serial
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>

#define BLYNK_TEMPLATE_ID "TMPL3I_BX0hNs"
#define BLYNK_TEMPLATE_NAME "Agriculture Robot"

char auth[] = "IxAL36i1k0LafHEnyTvnVipyinaKn1jb";  
char ssid[] = "Epik wifi"; 
char pass[] = "stonkstonks";  

const int analogInPin = A0;
 int sensorValue = 0;
 unsigned long int avgValue;
 float b;
 int buf[10], temp;
 
void setup()
{     
  Serial.begin(9600);
  Blynk.begin(auth, ssid, pass);    
}

void loop()
{
for (int i = 0; i < 10; i++) {
 buf[i] = analogRead(analogInPin);
 delay(10);
 }
 for (int i = 0; i < 9; i++)
 {
 for (int j = i + 1; j < 10; j++)
 {
 if (buf[i] > buf[j])
 {
 temp = buf[i];
 buf[i] = buf[j];
 buf[j] = temp; 
 }
 }
 }
 avgValue = 0;
 for (int i = 2; i < 8; i++)
 avgValue += buf[i];
 float pHVol = (float)avgValue * 5.0 / 1024 / 6;
 float phValue = -5.70 * pHVol + 7.34;
 Serial.println("sensor = ");
 Serial.print(phValue);
 Blynk.virtualWrite(V2,phValue);
 Blynk.run();
 delay(1000);
 }
