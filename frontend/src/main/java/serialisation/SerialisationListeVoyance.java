package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import metier.data.Client;
import metier.data.Voyance;

public class SerialisationListeVoyance extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        Client client = (Client) request.getSession().getAttribute("personne");
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();
        JsonArray listeVoyanceJson = new JsonArray();

        for (Voyance voyance : client.getListVoyance()) {

            JsonObject voyanceJson = new JsonObject();
            if (voyance.getDateDebut() != null) {
                voyanceJson.addProperty("medium", voyance.getMedium().getNom());
                SimpleDateFormat dateFormatter = new SimpleDateFormat("dd/MM/yyyy à HH:mm");
                String dateDeb = dateFormatter.format(voyance.getDateDebut());
                voyanceJson.addProperty("dateDebut", dateDeb);
                if (voyance.getDateFin() != null) {
                    String dateFin = dateFormatter.format(voyance.getDateFin());
                    voyanceJson.addProperty("dateFin", dateFin);
                } else {
                    voyanceJson.addProperty("dateFin", "En cours");
                }
                listeVoyanceJson.add(voyanceJson);
            }
        }

        jsonContainer.add("listeVoyance", listeVoyanceJson);
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(jsonContainer);
        out.println(json);
    }
}
