package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import metier.data.Client;

public class SerialisationClient extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();
        HttpSession session = request.getSession();
        Client client = (Client) session.getAttribute("personne");

        jsonContainer.addProperty("nom", client.getNom());
        jsonContainer.addProperty("prenom", client.getPrenom());
        jsonContainer.addProperty("couleur", client.getCouleur());
        jsonContainer.addProperty("zodiaque", client.getSigneZodiaque());
        jsonContainer.addProperty("chinois", client.getSigneChinois());
        jsonContainer.addProperty("animal", client.getAnimal());
        jsonContainer.addProperty("disponible", !client.getDemandeFaite());

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(jsonContainer);
        out.println(json);
    }

}
