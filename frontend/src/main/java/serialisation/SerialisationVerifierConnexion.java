package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

import metier.data.Client;
import metier.data.Employe;
import metier.data.Personne;

import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

public class SerialisationVerifierConnexion extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();
        HttpSession session = request.getSession();
        Personne personne = (Personne)session.getAttribute("personne");
        if (personne != null) {
            jsonContainer.addProperty("connexion", true);
            if (personne instanceof Client) {
                jsonContainer.addProperty("personne", "client");
            } else if (personne instanceof Employe) {
                jsonContainer.addProperty("personne", "employe");
            }
        } else {
            jsonContainer.addProperty("connexion", false);
        }
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(jsonContainer);
        out.println(json);
    }
}
