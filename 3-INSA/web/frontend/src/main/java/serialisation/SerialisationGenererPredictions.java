package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import metier.data.Employe;

public class SerialisationGenererPredictions extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();
        
        List<String> listePredictions = (List<String>) request.getAttribute("liste-predictions");

        jsonContainer.addProperty("predAmour", listePredictions.get(0));
        jsonContainer.addProperty("predSante", listePredictions.get(1));
        jsonContainer.addProperty("predTravail", listePredictions.get(2));

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(jsonContainer);
        out.println(json);
    }

}
