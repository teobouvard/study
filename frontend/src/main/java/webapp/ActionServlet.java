package webapp;

import serialisation.SerialisationStatistiques;
import action.ActionStatistiques;
import action.ActionValiderVoyance;
import serialisation.SerialisationValiderVoyance;
import action.*;
import serialisation.*;
import dao.JpaUtil;
import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import action.ActionPrendreRDV;

@WebServlet(name = "ActionServlet", urlPatterns = {"/ActionServlet"})
public class ActionServlet extends HttpServlet {

    @Override
    public void init() throws ServletException {
        super.init();
        JpaUtil.init();
    }

    @Override
    public void destroy() {
        JpaUtil.destroy();
        super.destroy();
    }

    /**
     * Processes requests for both HTTP <code>GET</code> and <code>POST</code>
     * methods.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    protected void processRequest(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        String todo = request.getParameter("todo");
        Action action = null;
        Serialisation serialisation = null;
        HttpSession session = request.getSession();

        switch (todo) {
            case "connecter":
                action = new ActionLogin();
                serialisation = new SerialisationLogin();
                break;

            case "inscrire":
                action = new ActionInscrire();
                serialisation = new SerialisationInscrire();
                break;
            case "deconnecter":
                response.sendRedirect(request.getContextPath() + "/index.html");
                session.invalidate();
                session = null;
                //action = new ActionDeconnecter();
                //serialisation = new SerialisationDeconnecter();
                //action.executer(request);
                //serialisation.serialiser(request, response);
                break;
            case "verifier-connexion":
                serialisation = new SerialisationVerifierConnexion();
                break;
        }
        if (session != null && session.getAttribute("personne") != null) {
            switch (todo) {
                case "infos-client":
                    serialisation = new SerialisationClient();
                    break;
                case "infos-employe":
                    serialisation = new SerialisationEmploye();
                    break;
                case "liste-medium":
                    action = new ActionListerMediums();
                    serialisation = new SerialisationListerMedium();
                    break;
                case "infos-medium":
                    action = new ActionInfosMedium();
                    serialisation = new SerialisationInfosMedium();
                    break;
                case "historique-client":
                    serialisation = new SerialisationListeVoyance();
                    break;
                case "voyance-en-cours":
                    action = new ActionChercherVoyance();
                    serialisation = new SerialisationVoyance();
                    break;
                case "prendre-rdv":
                    action = new ActionPrendreRDV();
                    serialisation = new SerialisationRDV();
                    break;
                case "demarrer-voyance":
                    action = new ActionDemarrerVoyance();
                    serialisation = new SerialisationDemarrerVoyance();
                    break;
                case "generer-predictions":
                    action = new ActionGenererPredictions();
                    serialisation = new SerialisationGenererPredictions();
                    break;
                case "valider-voyance":
                    action = new ActionValiderVoyance();
                    serialisation = new SerialisationValiderVoyance();
                    break;
                case "statistiques":
                    action = new ActionStatistiques();
                    serialisation = new SerialisationStatistiques();
                    break;
            }
        }
        if(action != null)
        {
            action.executer(request);
        }
        if(serialisation != null)
        {
            serialisation.serialiser(request, response);
        }
    }

    // <editor-fold defaultstate="collapsed" desc="HttpServlet methods. Click on the + sign on the left to edit the code.">
    /**
     * Handles the HTTP <code>GET</code> method.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        processRequest(request, response);
    }

    /**
     * Handles the HTTP <code>POST</code> method.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        processRequest(request, response);
    }

    /**
     * Returns a short description of the servlet.
     *
     * @return a String containing servlet description
     */
    @Override
    public String getServletInfo() {
        return "Short description";
    }// </editor-fold>

}
