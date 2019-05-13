package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import metier.data.Astrologue;
import metier.data.Medium;
import metier.data.Tarologue;
import metier.data.Voyant;

public class SerialisationInfosMedium extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        Medium medium = (Medium) request.getAttribute("medium");
        JsonObject mediumJson = new JsonObject();

        mediumJson.addProperty("nom", medium.getNom());
        mediumJson.addProperty("description", medium.getDescriptif());
        mediumJson.addProperty("id", medium.getId());
        if (medium instanceof Astrologue) {
            mediumJson.addProperty("type", "Astrologue");
            mediumJson.addProperty("promotion", ((Astrologue) medium).getPromotion());
            mediumJson.addProperty("formation", ((Astrologue) medium).getFormation());
        } else if (medium instanceof Tarologue) {
            mediumJson.addProperty("type", "Tarologue");
        } else if (medium instanceof Voyant) {
            mediumJson.addProperty("type", "Voyante");
            mediumJson.addProperty("specialite", ((Voyant) medium).getSpecialite());
        }

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(mediumJson);
        out.println(json);
    }

}
