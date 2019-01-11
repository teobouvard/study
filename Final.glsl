const int Steps = 1000;
const float Epsilon = 0.01; // Marching epsilon
const float T=0.5;

const float rA=1.0; // Minimum ray marching distance from origin
const float rB=50.0; // Maximum

vec2 hash( vec2 p ) 
{
	p = vec2( dot(p,vec2(127.1,311.7)),
			  dot(p,vec2(269.5,183.3)) );

	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec2 p)
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2 i = floor( p + (p.x+p.y)*K1 );
	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = step(a.yx,a.xy);    
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;

    vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );

	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));

    return dot( n, vec3(70) );
	
}

float ridged(vec3 p)
{
    float result = 2.0*(0.5-abs(0.5-noise(p.xz)));
    return result;
}

float turbulence(vec3 p, float ampl, float f0, float nbOctaves, float fA)
{
    float result = 0.0;
    for(float i = 0.0; i<nbOctaves;i = i + 1.0){
        result = result + ampl*ridged(f0*p);
        ampl = ampl*fA;
        f0 = 2.0*f0;
    } 
    return result;
    
}

float Terrain(vec3 p, out int indiceObj)
{
    //float new_y = noise(p.xz,70.0,0.5);
    
    indiceObj = 1;
    
    float new_y = turbulence(p,0.7,0.4,5.0,0.39);
    
    if(new_y < -0.2){
        new_y = -0.2 + 0.1*turbulence(p,0.7,0.4,5.0,0.39)*sin(3.0*iTime+p.x*p.z);
        indiceObj=2;
    }
    float terrain = new_y - p.y;    
    return terrain;
    
}


// Transforms
vec3 rotateY(vec3 p, float a)
{
   float new_x = p.x*cos(a) - p.z*sin(a);
   float new_z = p.x*sin(a) + p.z*cos(a);
   return vec3(new_x,p.y, new_z);

   return p;
}

// Smooth falloff function
// r : small radius
// R : Large radius
float falloff( float r, float R )
{
   float x = clamp(r/R,0.0,1.0);
   float y = (1.0-x*x);
   return y*y*y;
}

// Primitive functions

// Point skeleton
// p : point
// c : center of skeleton
// e : energy associated to skeleton
// R : large radius
float point(vec3 p, vec3 c, float e,float R)
{
   return e*falloff(length(p-c),R);
}


// Blending
// a : field function of left sub-tree
// b : field function of right sub-tree
float Blend(float a,float b)
{
   return a+b;
}


// Potential field of the object
// p : point
float object(vec3 p, out int indiceObj)
{
   //float v = Blend(point(p,vec3( -2.5, 0.0,0.0),1.0,4.5),
                  // point(p,vec3( 2.5, 0.0,0.0),1.0,4.5));
    
    float v = Terrain(p,indiceObj);
    

   return v-T;
}



// Calculate object normal
// p : point
vec3 ObjectNormal(in vec3 p, out int indiceObj)
{
   float eps = 0.0001;
   vec3 n;
   float v = object(p,indiceObj);
   n.x = object( vec3(p.x+eps, p.y, p.z) ,indiceObj) - v;
   n.y = object( vec3(p.x, p.y+eps, p.z) ,indiceObj) - v;
   n.z = object( vec3(p.x, p.y, p.z+eps) ,indiceObj) - v;
   return normalize(n);
}

// Trace ray using ray marching
// o : ray origin
// u : ray direction
// h : hit
// s : Number of steps
float Trace(vec3 o, vec3 u, out bool h,out int s,out int indiceObj)
{
   h = false;

   // Don't start at the origin
   // instead move a little bit forward
   float t=rA;

   for(int i=0; i<Steps; i++)
   {
      s=i;
      vec3 p = o+t*u;
      float v = object(p,indiceObj);
      // Hit object (1) 
      if (v > 0.0)
      {
         s=i;
         h = true;
         break;
      }
      // Move along ray
      t += Epsilon;  

      // Escape marched far away
      if (t>rB)
      {
         break;
      }
   }
   return t;
}

// Background color
vec3 background(vec3 rd)
{
    vec3 color;
    color = mix(vec3(0.8, 0.8, 0.9), vec3(0.6, 0.9, 1.0), rd.y*1.0+0.25);
    
    //Soleil
     if(sqrt((rd.x-0.01*iTime)*(rd.x-0.01*iTime)+(rd.y+0.01*iTime-0.2)*(rd.y+0.01*iTime-0.2))<0.1){
         color = vec3(1,(170.0-0.9*iTime)/255.0,0) + turbulence(rd,0.10,0.5,5.0,0.7);
     }
    
	
    if (iTime <20.0){
         
        //Nuage 
    
        if(rd.y>0.1 && rd.y<0.15){
            if(mod(0.01*iTime+rd.x, 0.5) < 0.08){
                color = vec3(1,1,1)-iTime*0.01;
            }
            
        }
         
    } else if(iTime>20.0 && iTime <30.0){
        color = color - 0.01*(iTime-20.0)*(iTime-20.0) ;
        
        //Nuage jour
        if(rd.y>0.1 && rd.y<0.15){
            if(mod(0.01*iTime+rd.x, 0.5) < 0.08){
                color = vec3(1,1,1)-iTime*0.01;
            }
    	}
    } else if(iTime>30.0){
        //Lune
        color = mix(vec3(0.8, 0.8, 0.9), vec3(0.6, 0.9, 1.0), rd.y*1.0+0.25) - 0.01*(iTime-20.0)*(iTime-20.0) ;
        if(sqrt(pow(rd.x+0.8-0.01*(iTime-5.0),2.0)+pow((rd.y+0.3-0.01*(iTime-5.0)),2.0))<0.05){
        	color = vec3(1,1,1);
        }
        //Nuage nuit
        if(rd.y>0.1 && rd.y<0.15){
            if(mod(0.01*iTime+rd.x, 0.5) < 0.08){
                if(1.0-iTime*0.01>0.1){    
                	color = vec3(1,1,1)-iTime*0.01;
                } else {
                    color = vec3(0.2,0.2,0.2);
                }
            }
    	}
    }
    
    
    
   return color;
}

// Shading and lighting
// p : point,
// n : normal at point
vec3 Shade(vec3 p, vec3 n, int s, int indiceObj)
{
   // point light
   const vec3 lightPos = vec3(5.0, 5.0, 5.0);
   const vec3 lightColor = vec3(1.0, 1.0, 1.0);

   vec3 l = normalize(lightPos - p);

   // Not even Phong shading, use weighted cosine instead for smooth transitions
   float diff = 0.5*(1.0+dot(n, l));

   vec3 c =  0.5*vec3(0.5,0.5,0.5)+0.5*diff*lightColor;
   float fog = 0.7*float(s)/(float(Steps-1));
   c = (1.0-fog)*c+fog*vec3(1.0,1.0,1.0);
    
    //Couleurs souhaitées en haut et bas de montagne
    
    vec3 c_bas = vec3(43.0/255.0,21.0/255.0,3.0/255.0);
    vec3 c_haut = vec3(160.0/255.0,143.0/255.0,101.0/255.0);
    
    //Redimensionnement de p_y_temp pour qu'il soit compris entre 0 et 1
    
    float p_y_temp = (p.y+1.0)/2.0;
    
    //Affectation de la couleur en fonction de l'altitude
    if(indiceObj==1){
        c =   c/4.0 + c_haut*(p_y_temp-0.2)*1.3 + c_bas*(1.0-(p_y_temp-0.2))  + turbulence(c,0.10,0.4,3.0,0.60);
        //Lignes noires
        if(mod(p.y,.1)<0.01){
        c = vec3(0.0,0.0,0.0);
    	}
    } else if(indiceObj==2){
        //Lac
        c = vec3(0.0/255.0,107.0/255.0,179.0/255.0)+turbulence(c,0.10,0.5,5.0,0.7);
    } 
    
    
    
    

   return c;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord)
{
   vec2 pixel = (gl_FragCoord.xy / iResolution.xy)*2.0-1.0;

   // compute ray origin and direction
   float asp = iResolution.x / iResolution.y;
   vec3 rd = vec3(asp*pixel.x, pixel.y, -4.0);
   vec3 ro = vec3(0.0, 0.0, 15.0);

   vec2 mouse = iMouse.xy / iResolution.xy;
   float a=-mouse.x;//iTime*0.25;
   rd.z = rd.z+2.0*mouse.y;
   rd = normalize(rd);
   ro = rotateY(ro, a);
   rd = rotateY(rd, a);

   // Trace ray
   bool hit;

   // Number of steps
   int s;
   int indiceObj;
   float t = Trace(ro, rd, hit,s, indiceObj);
   vec3 pos=ro+t*rd;
   // Shade background
   vec3 rgb = background(rd);

   if (hit)
   {
      // Compute normal
      
      vec3 n = ObjectNormal(pos, indiceObj);

      // Shade object with light
      rgb = Shade(pos, n, s, indiceObj);
   }
   
    
   	fragColor=vec4(rgb, 1.0);

}


