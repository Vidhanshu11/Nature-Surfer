/* tempature, humidity and ldr  */
#include <Blynk.h>
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>
#include <DHT.h>

/---dht---/
#define DHTPIN D4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define BLYNK_TEMPLATE_ID "TMPL3I_BX0hNs"
#define BLYNK_TEMPLATE_NAME "Agriculture Robot"
#define BLYNK_AUTH_TOKEN "IxAL36i1k0LafHEnyTvnVipyinaKn1jb"

char auth[] = "IxAL36i1k0LafHEnyTvnVipyinaKn1jb";
char ssid[] = "Epik wifi";
char pass[] = "stonkstonks";

BlynkTimer timer;
#define LDRInput A0

 //Set Analog Input A0 for LDR.
void setup() {
Serial.begin(115200);
Blynk.begin(auth, ssid, pass, "blynk.cloud", 80);
pinMode(LDRInput,INPUT);
pinMode(LED_BUILTIN,OUTPUT);
dht.begin();
timer.setInterval(1000L, sendSensor);
}

void loop() {

  int value=analogRead(LDRInput);//Reads the Value of LDR(light).
  float hum = dht.readHumidity();
  float temp = dht.readTemperature(); 
 
  Serial.println("LDR value is :");//Prints the value of LDR to Serial Monitor.
  Serial.println(value);
  delay(500);
  Blynk.virtualWrite(V5, value);
  Blynk.virtualWrite(V1, hum);
  Blynk.virtualWrite(V0, temp);
  Blynk.run();
}

void sendSensor(){
 float hum = dht.readHumidity();
 float temp = dht.readTemperature(); 
 
 if (isnan(hum) || isnan(temp)) {
 Serial.println("Failed to read from DHT sensor!");
 return;
 }
 Serial.print("T: ");
 Serial.println(temp);
 Serial.println(hum);
 Blynk.virtualWrite(V0, hum);
 Blynk.run();
}
