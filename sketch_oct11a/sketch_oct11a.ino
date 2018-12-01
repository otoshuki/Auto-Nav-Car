const int powerleft = 5;
const int powerright = 6;
const int dir_left1 = 2;
const int dir_left2 = 4;
const int dir_right1 = 8;
const int dir_right2 = 7;

int data[4];
int speed_l,speed_r,dir_l,dir_r;

void setup()
{
  // put your setup code here, to run once:

  pinMode(powerleft, OUTPUT); 
  pinMode(powerright, OUTPUT);
  pinMode(dir_left1, OUTPUT);
  pinMode(dir_left2, OUTPUT);
  pinMode(dir_right1, OUTPUT);
  pinMode(dir_right2, OUTPUT);
  Serial.begin(9600);

}

void loop() 
{ 
  // put your main code here, to run repeatedly:

  if (Serial.available() >= 4)
    {
       //Take input data
       for (int i = 0; i<4; i++)
       {
        data[i] = Serial.read();
       }
       //Set the dir and speed values
       dir_l = data[0];
       speed_l = data[1] * 256;
       dir_r = data[2];
       speed_r = data[3] * 256;

       //Print
       Serial.println("Direction_left = ");
       Serial.print(dir_l);
       Serial.print('\n');
       Serial.println("speedleft = "); 
       Serial.print(speed_l);
       Serial.print('\n');
       Serial.println("Direction_right = ");
       Serial.print(dir_r);
       Serial.print('\n');
       Serial.println("speedright = ");
       Serial.print(speed_r);
       Serial.print('\n');
    
       if ( dir_l == 0 )
         {
  
              analogWrite(powerleft, speed_l);
              digitalWrite(dir_left1, LOW);
              digitalWrite(dir_left2, HIGH);   
         }
       else if (dir_l == 1)
         {
              analogWrite(powerleft, speed_l);
              digitalWrite(dir_left1, HIGH);
              digitalWrite(dir_left2, LOW);  
         }

       if ( dir_r == 0)
         {
               analogWrite(powerright, speed_r);
               digitalWrite(dir_right1, LOW);
               digitalWrite(dir_right2, HIGH);   
         }
       else if (dir_r == 1)
       {
               analogWrite(powerright, speed_r);
               digitalWrite(dir_right1, HIGH);
               digitalWrite(dir_right2, LOW);  
       }
      delay(500);
     }     
}
