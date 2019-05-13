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

public class SerialisationVoyance extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();
        if (request.getAttribute("statut") == Boolean.TRUE) {

            jsonContainer.addProperty("statut", true);
            Voyance voyance = (Voyance) request.getAttribute("voyance");
            Client client = voyance.getClient();
            jsonContainer.addProperty("nomClient", client.getNom());
            jsonContainer.addProperty("prenomClient", client.getPrenom());
            jsonContainer.addProperty("medium", voyance.getMedium().getNom());
            jsonContainer.addProperty("couleur", client.getCouleur());
            jsonContainer.addProperty("zodiaque", client.getSigneZodiaque());
            jsonContainer.addProperty("chinois", client.getSigneChinois());
            jsonContainer.addProperty("animal", client.getAnimal());

            JsonArray listeVoyanceJson = new JsonArray();
            for (Voyance voyanceIterator : client.getListVoyance()) {

                JsonObject voyanceJson = new JsonObject();
                if (voyanceIterator.getDateDebut() != null) {
                    voyanceJson.addProperty("medium", voyanceIterator.getMedium().getNom());
                    SimpleDateFormat dateFormatter = new SimpleDateFormat("dd/MM/yyyy à HH:mm");
                    String dateDeb = dateFormatter.format(voyanceIterator.getDateDebut());
                    voyanceJson.addProperty("dateDebut", dateDeb);
                    if (voyanceIterator.getDateFin() != null) {
                        String dateFin = dateFormatter.format(voyanceIterator.getDateFin());
                        voyanceJson.addProperty("dateFin", dateFin);
                    } else {
                        voyanceJson.addProperty("dateFin", "En cours");
                    }
                    listeVoyanceJson.add(voyanceJson);
                }
            }
            jsonContainer.add("listeVoyance", listeVoyanceJson);
        } else {
            jsonContainer.addProperty("statut", false);
        }

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(jsonContainer);
        out.println(json);
    }
}
