/*code for relay*/

#include <Blynk.h>
#define BLYNK_PRINT Serial
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>

#define BLYNK_TEMPLATE_ID "TMPL3I_BX0hNs"
#define BLYNK_TEMPLATE_NAME "Agriculture Robot"

char auth[] = "IxAL36i1k0LafHEnyTvnVipyinaKn1jb";  
char ssid[] = "Epik wifi"; 
char pass[] = "stonkstonks";

int relaypin = D2;  //GPIO4
int relaypin1 = D6;  //GPIO14

void setup() {
  Serial.begin(9600);
  Blynk.begin(auth, ssid, pass);  
  pinMode(relaypin,OUTPUT);
  pinMode(relaypin1,OUTPUT);  
}

void loop() {
  Blynk.run();
  delay(1000);
}
