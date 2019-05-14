package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class SerialisationInscrire extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();
        Boolean statut = (Boolean) request.getAttribute("statut");

        if (statut) {
            jsonContainer.addProperty("inscription", true);

        } else {
            jsonContainer.addProperty("inscription", false);
        }
        
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(jsonContainer);
        out.println(json);
    }
}
