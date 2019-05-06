package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import metier.data.Astrologue;
import metier.data.Medium;
import metier.data.Tarologue;
import metier.data.Voyant;
import serialisation.Serialisation;

public class SerialisationListerMedium extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();
        List<Medium> listeMedium = (List<Medium>) request.getAttribute("liste-medium");
        JsonArray listeMediumJson = new JsonArray();
        for (Medium medium : listeMedium) {
            JsonObject mediumJson = new JsonObject();
            
            mediumJson.addProperty("nom", medium.getNom());
            mediumJson.addProperty("description", medium.getDescriptif());
            mediumJson.addProperty("id", medium.getId());
            if(medium instanceof Astrologue){
                mediumJson.addProperty("type","Astrologue");
                mediumJson.addProperty("promotion",((Astrologue)medium).getPromotion());
                mediumJson.addProperty("formation",((Astrologue)medium).getFormation());
            } else if (medium instanceof Tarologue){
                mediumJson.addProperty("type","Tarologue");
            } else if (medium instanceof Voyant){
                mediumJson.addProperty("type","Voyant");
                mediumJson.addProperty("specialite",((Voyant)medium).getSpecialite());
            }
 
            listeMediumJson.add(mediumJson);
        }
        jsonContainer.add("listemedium", listeMediumJson);
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(jsonContainer);
        out.println(json);
    }

}
